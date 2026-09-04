"""
O estudo declarativo: :class:`SurvivalConfig` → :func:`run_survival_study`
=========================================================================
A interface e o pipeline falam a **mesma língua**: a configuração que gerou uma
curva viaja com ela em JSON e reproduz o resultado fora do notebook. Este módulo
é o análogo, para o eixo *lifetime*, de
:class:`~yggdrasil.credit_risk.econometric.config.StudyConfig`/``run_study``.

O estudo encadeia:

1. **Partição** DES/OOT do painel (:func:`split_panel`): por safra de
   originação, por data de observação, por uma coluna de amostra, ou nenhuma;
2. **Curva** no DES: Kaplan-Meier ou safra (por grupo, se ``by``), regressão de
   *hazard* com covariáveis, ou família paramétrica pura;
3. **Cauda**: para as curvas empíricas, a extrapolação além da última idade com
   base suficiente vem da família paramétrica escolhida (ou é plana);
4. **Ajustes**: calibração do nível à PD de 12 meses do modelo transversal e o
   condicionamento ao ciclo (Vasicek);
5. **Validação** no OOT (ou no próprio DES, sinalizado): backtest por horizonte,
   discriminação, C-index, calibração por decil e, no motor de *hazard*, o teste
   de riscos proporcionais.

O resultado (:class:`SurvivalResult`) carrega o modelo final
(:class:`~yggdrasil.credit_risk.ecl.lifetime_pd.LifetimePD`, pronto para
``apply`` e para o ECL), as tabelas de cada etapa e o tempo de cada uma.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..ecl.curves import PDCurve, vintage_curve
from ..ecl.lifetime_pd import LifetimePD
from ..ecl.panel import ContractPanel
from ..ecl.survival import DiscreteHazard, kaplan_meier
from .lifetable import GLOBAL, life_table, logrank_test
from .parametric import DISTRIBUTIONS, ParametricSurvival, junction_age, splice_curves
from .validation import (
    backtest_curve,
    calibration_by_decile,
    discrimination_by_horizon,
    model_concordance,
    ph_test,
)

#: Motores de curva do estudo.
STUDY_METHODS = ("km", "vintage", "hazard", "parametric")

#: Modos de partição DES/OOT.
SPLIT_MODES = ("none", "origin", "observation", "column")

#: Tratamento da cauda além da última idade confiável.
TAILS = ("parametric", "flat")


# ======================================================================
# Configuração
# ======================================================================
@dataclass
class SurvivalConfig:
    """Configuração completa de um estudo de sobrevivência para PD *lifetime*.

    Os campos de coluna descrevem o painel (os mesmos do
    :class:`~yggdrasil.credit_risk.ecl.panel.ContractPanel`); os demais, cada
    etapa do estudo. Serializa em JSON (:meth:`to_json`) e volta
    (:meth:`from_json`).
    """

    name: str = "estudo_sobrevivencia"
    # --- o painel ---------------------------------------------------------
    id_col: str = "id_contrato"
    date_col: str = "dt_ref"
    default_col: str = "default"
    age_col: Optional[str] = None
    origin_col: Optional[str] = None
    term_col: Optional[str] = None
    segment_col: Optional[str] = None
    exposure_col: Optional[str] = None
    freq: str = "M"
    # --- a curva -----------------------------------------------------------
    method: str = "km"
    by: Optional[str] = None
    horizon: int = 60
    from_age: int = 0
    min_at_risk: int = 30
    alpha: float = 0.05
    weighted: bool = False
    # --- hazard com covariáveis --------------------------------------------
    features: List[str] = field(default_factory=list)
    baseline: str = "spline"
    link: str = "logit"
    C: float = 1e6
    n_knots: int = 6
    max_age: Optional[int] = None
    # --- cauda / paramétrico -----------------------------------------------
    tail: str = "parametric"
    distribution: str = "weibull"
    junction: Optional[int] = None
    match_level: bool = True
    # --- calibração e ciclo -------------------------------------------------
    calibrate: Dict[str, float] = field(default_factory=dict)
    z: Optional[float] = None
    rho: float = 0.10
    decay: Optional[float] = None
    mode: str = "shift"
    # --- validação ------------------------------------------------------------
    split: str = "none"
    split_value: Optional[str] = None
    split_col: Optional[str] = None
    oot_value: str = "OOT"
    backtest_horizons: List[int] = field(default_factory=lambda: [12, 24, 36])
    decile_horizon: int = 12
    n_bins: int = 10

    def __post_init__(self) -> None:
        if self.method not in STUDY_METHODS:
            raise ValueError(f"method deve ser um de {STUDY_METHODS}; recebido {self.method!r}.")
        if self.split not in SPLIT_MODES:
            raise ValueError(f"split deve ser um de {SPLIT_MODES}; recebido {self.split!r}.")
        if self.tail not in TAILS:
            raise ValueError(f"tail deve ser um de {TAILS}; recebido {self.tail!r}.")
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(
                f"distribution deve ser uma de {DISTRIBUTIONS}; recebido {self.distribution!r}.")
        if int(self.horizon) < 1:
            raise ValueError(f"horizon deve ser >= 1; recebido {self.horizon!r}.")
        if self.method == "hazard" and not self.features:
            raise ValueError("method='hazard' exige features.")
        self.features = list(self.features)
        self.backtest_horizons = [int(h) for h in self.backtest_horizons]
        self.calibrate = {str(k): float(v) for k, v in (self.calibrate or {}).items()}

    # -- serialização --------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "SurvivalConfig":
        campos = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in dict(d).items() if k in campos})

    def to_json(self, path: Optional[str] = None) -> str:
        txt = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(txt)
        return txt

    @classmethod
    def from_json(cls, path_or_text: str) -> "SurvivalConfig":
        texto = path_or_text
        if not str(path_or_text).lstrip().startswith("{"):
            with open(path_or_text, "r", encoding="utf-8") as fh:
                texto = fh.read()
        return cls.from_dict(json.loads(texto))

    def panel_kwargs(self) -> dict:
        """Os argumentos de coluna para montar o ``ContractPanel``."""
        return dict(id_col=self.id_col, date_col=self.date_col, default_col=self.default_col,
                    age_col=self.age_col, origin_col=self.origin_col, term_col=self.term_col,
                    segment_col=self.segment_col, exposure_col=self.exposure_col,
                    freq=self.freq)

    def build_panel(self, df: pd.DataFrame) -> ContractPanel:
        """Monta o painel a partir do DataFrame longo com as colunas desta config."""
        return ContractPanel(df, **self.panel_kwargs())


# ======================================================================
# Partição DES / OOT
# ======================================================================
def _sub_panel(panel: ContractPanel, mask) -> Optional[ContractPanel]:
    parte = panel.df[np.asarray(mask, dtype=bool)]
    if parte.empty:
        return None
    return ContractPanel(parte.reset_index(drop=True), id_col=panel.id_col,
                         date_col=panel.date_col, default_col=panel.default_col,
                         age_col=panel.age_col, origin_col=None, term_col=panel.term_col,
                         segment_col=panel.segment_col, exposure_col=panel.exposure_col,
                         freq=panel.freq, drop_post_default=False)


def split_panel(panel: ContractPanel, mode: str = "none", value=None,
                column: Optional[str] = None, oot_value="OOT"
                ) -> Tuple[ContractPanel, Optional[ContractPanel]]:
    """Parte o painel em ``(DES, OOT)``.

    ``mode``:

    * ``'none'``: tudo é DES; OOT é ``None``;
    * ``'origin'``: contratos **originados** antes de ``value`` (data) vão para o
      DES, os demais para o OOT (a validação por safra, a mais exigente: o OOT
      são contratos que o ajuste nunca viu);
    * ``'observation'``: observações **anteriores** a ``value`` formam o DES e as
      posteriores o OOT (mesmos contratos, janela de tempo distinta: o OOT começa
      em idades altas e testa a curva a partir dali);
    * ``'column'``: ``column == oot_value`` é OOT; o resto é DES.
    """
    if mode not in SPLIT_MODES:
        raise ValueError(f"mode deve ser um de {SPLIT_MODES}; recebido {mode!r}.")
    if mode == "none":
        return panel, None
    d = panel.df
    if mode == "column":
        if not column or column not in d.columns:
            raise ValueError(f"split por coluna exige uma coluna existente; recebido {column!r}.")
        oot = d[column].astype(str) == str(oot_value)
    else:
        if value is None:
            raise ValueError(f"split='{mode}' exige a data de corte em value.")
        corte = pd.Timestamp(value)
        if mode == "observation":
            oot = pd.to_datetime(d[panel.date_col]) >= corte
        else:
            if panel.origin_col and panel.origin_col in d.columns:
                origem = pd.to_datetime(d[panel.origin_col])
            else:
                origem = d.groupby(panel.id_col, sort=False)[panel.date_col].transform("min")
            oot = pd.to_datetime(origem) >= corte
    des = _sub_panel(panel, ~oot)
    if des is None:
        raise ValueError("a partição deixou o DES vazio; revise o corte.")
    return des, _sub_panel(panel, oot)


# ======================================================================
# Resultado
# ======================================================================
@dataclass
class SurvivalResult:
    """Tudo o que o estudo produziu.

    Attributes
    ----------
    config:
        A configuração que gerou o resultado.
    panel_des, panel_oot:
        Os painéis da partição (``panel_oot`` é ``None`` sem partição).
    life_table:
        Tabela de vida do DES (por grupo, se ``by``).
    logrank:
        Resultado do log-rank entre os grupos de ``by`` (ou ``None``).
    model_base:
        O :class:`LifetimePD` **antes** de calibração e ciclo.
    model:
        O :class:`LifetimePD` final (com cauda, calibração e ciclo aplicados).
    parametric:
        ``{grupo: ParametricSurvival}`` das caudas ajustadas (ou ``None``).
    hazard_models:
        ``{grupo: DiscreteHazard}`` quando ``method='hazard'``.
    backtest, discrimination, decile, c_index, ph:
        As saídas da validação (no OOT quando há partição; senão no DES).
    validated_on:
        ``'OOT'`` ou ``'DES (in-sample)'``.
    steps:
        Lista de ``(etapa, segundos)``.
    """

    config: SurvivalConfig
    panel_des: ContractPanel
    panel_oot: Optional[ContractPanel]
    life_table: pd.DataFrame
    logrank: Optional[dict]
    model_base: LifetimePD
    model: LifetimePD
    parametric: Optional[Dict[object, ParametricSurvival]]
    hazard_models: Dict[object, DiscreteHazard]
    backtest: Optional[pd.DataFrame]
    discrimination: Optional[pd.DataFrame]
    decile: Optional[pd.DataFrame]
    c_index: Optional[float]
    ph: Optional[dict]
    validated_on: str
    steps: List[Tuple[str, float]] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """Uma linha por curva do modelo final, com o resultado do backtest anexo."""
        s = self.model.summary()
        if self.backtest is not None and len(self.backtest):
            bt = self.backtest.groupby("grupo")["dentro_do_ic"].mean().rename("pct_dentro_do_ic")
            s = s.merge(bt, left_on="grupo", right_index=True, how="left")
        return s

    def to_dict(self) -> dict:
        return {"config": self.config.to_dict(), "model": self.model.to_dict(),
                "validated_on": self.validated_on,
                "c_index": self.c_index,
                "steps": [{"etapa": e, "segundos": s} for e, s in self.steps]}


# ======================================================================
# O estudo
# ======================================================================
def _curves_por_grupo(panel: ContractPanel, cfg: SurvivalConfig) -> Dict[object, PDCurve]:
    partes = {GLOBAL: panel} if cfg.by is None else panel.by(cfg.by)
    out = {}
    for rot, parte in partes.items():
        rotulo = "" if rot == GLOBAL else str(rot)
        if cfg.method == "vintage":
            out[rot] = vintage_curve(parte, from_age=cfg.from_age, weighted=cfg.weighted,
                                     min_at_risk=1, fill="ffill", label=rotulo,
                                     alpha=cfg.alpha)
        else:
            out[rot] = kaplan_meier(parte, from_age=cfg.from_age, weighted=cfg.weighted,
                                    alpha=cfg.alpha, label=rotulo)
    return out


def run_survival_study(source, config: SurvivalConfig,
                       progress: Optional[Callable[[str, str, str], None]] = None
                       ) -> SurvivalResult:
    """Roda o estudo completo a partir do DataFrame longo (ou de um painel).

    Parameters
    ----------
    source:
        ``pandas.DataFrame`` no formato longo (o painel é montado com as colunas
        da config) ou um :class:`ContractPanel` já pronto.
    config:
        A :class:`SurvivalConfig`.
    progress:
        Callback ``(chave, rótulo, status)`` chamado no início e no fim de cada
        etapa (``status`` em ``'run'``/``'ok'``); a interface usa para a tabela
        de progresso.
    """
    cfg = config
    steps: List[Tuple[str, float]] = []

    def _etapa(chave, rotulo):
        class _Ctx:
            def __enter__(self_inner):
                self_inner.t0 = time.monotonic()
                if progress:
                    progress(chave, rotulo, "run")
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                steps.append((rotulo, time.monotonic() - self_inner.t0))
                if progress and exc_type is None:
                    progress(chave, rotulo, "ok")
                return False
        return _Ctx()

    # 1. painel e partição --------------------------------------------------
    with _etapa("painel", "Painel e partição DES/OOT"):
        panel = source if isinstance(source, ContractPanel) else cfg.build_panel(source)
        des, oot = split_panel(panel, mode=cfg.split, value=cfg.split_value,
                               column=cfg.split_col, oot_value=cfg.oot_value)
        by_ok = cfg.by is not None and cfg.by in des.df.columns
        if cfg.by is not None and not by_ok:
            raise ValueError(f"coluna de grupo {cfg.by!r} não existe no painel.")

    # 2. tabela de vida e log-rank ------------------------------------------
    with _etapa("vida", "Tabela de vida e log-rank"):
        tab = life_table(des, by=cfg.by, from_age=cfg.from_age, alpha=cfg.alpha,
                         weighted=cfg.weighted)
        lr = None
        if cfg.by is not None and len(des.by(cfg.by)) >= 2 and not cfg.weighted:
            try:
                lr = logrank_test(des, cfg.by, from_age=cfg.from_age)
            except ValueError:
                lr = None

    # 3. a curva ----------------------------------------------------------------
    hazard_models: Dict[object, DiscreteHazard] = {}
    parametric: Optional[Dict[object, ParametricSurvival]] = None
    with _etapa("curva", f"Curva ({cfg.method})"):
        if cfg.method == "hazard":
            base = LifetimePD(method="hazard", horizon=cfg.horizon, baseline=cfg.baseline,
                              link=cfg.link, C=cfg.C, n_knots=cfg.n_knots, max_age=cfg.max_age
                              ).fit(des, by=cfg.by, features=cfg.features)
            hazard_models = dict(base.hazard_models_)
        elif cfg.method == "parametric":
            partes = {GLOBAL: des} if cfg.by is None else des.by(cfg.by)
            parametric, curvas = {}, {}
            for rot, parte in partes.items():
                m = ParametricSurvival(cfg.distribution).fit(parte, max_age=cfg.max_age,
                                                             min_at_risk=1)
                parametric[rot] = m
                curvas[rot] = m.curve(horizon=cfg.horizon, from_age=cfg.from_age,
                                      label="" if rot == GLOBAL else str(rot))
            base = LifetimePD.from_curves(curvas, method="km")
            base.by = cfg.by
            base.meta.update({"metodo": "parametric", "distribution": cfg.distribution})
        else:
            empiricas = _curves_por_grupo(des, cfg)
            partes = {GLOBAL: des} if cfg.by is None else des.by(cfg.by)
            curvas = {}
            if cfg.tail == "parametric":
                parametric = {}
            for rot, curva in empiricas.items():
                parte = partes[rot]
                if cfg.tail == "parametric":
                    j = cfg.junction if cfg.junction is not None else junction_age(
                        parte, min_at_risk=cfg.min_at_risk)
                    if j < cfg.from_age:
                        j = cfg.from_age
                    m = ParametricSurvival(cfg.distribution).fit(parte, max_age=cfg.max_age,
                                                                 min_at_risk=1)
                    parametric[rot] = m
                    curvas[rot] = splice_curves(curva, m, junction=j, horizon=cfg.horizon,
                                                match_level=cfg.match_level,
                                                from_age=cfg.from_age)
                else:
                    curvas[rot] = curva.extend(cfg.horizon)
            base = LifetimePD.from_curves(curvas, method=cfg.method)
            base.by = cfg.by
            base.meta.update({"metodo": cfg.method, "tail": cfg.tail,
                              "distribution": cfg.distribution if cfg.tail == "parametric" else None})
        base.freq = des.freq
        base.meta.update({"by": cfg.by, "horizon": cfg.horizon, "n_contratos": des.n_contracts,
                          "features": list(cfg.features)})

    # 4. ajustes ----------------------------------------------------------------
    with _etapa("ajustes", "Calibração e ciclo"):
        model = base
        if cfg.calibrate:
            alvos = {}
            for k, v in cfg.calibrate.items():
                chave = GLOBAL if k in ("", GLOBAL, "global") else k
                # as chaves do JSON são strings; as das curvas podem não ser
                for c in model.curves_:
                    if c == chave or str(c) == str(chave):
                        alvos[c] = float(v)
            if alvos:
                model = model.calibrate_to(alvos)
        if cfg.z is not None:
            model = model.condition(float(cfg.z), rho=float(cfg.rho), decay=cfg.decay,
                                    mode=cfg.mode)

    # 5. validação ----------------------------------------------------------------
    alvo_val = oot if oot is not None else des
    validated_on = "OOT" if oot is not None else "DES (in-sample)"
    backtest = discrimination = decile = None
    c_index = None
    ph = None
    with _etapa("validacao", f"Validação ({validated_on})"):
        try:
            backtest = backtest_curve(model, alvo_val, horizons=cfg.backtest_horizons,
                                      alpha=cfg.alpha)
        except ValueError:
            backtest = None
        try:
            discrimination = discrimination_by_horizon(model, alvo_val,
                                                      horizons=cfg.backtest_horizons)
        except (ValueError, KeyError):
            discrimination = None
        try:
            c_index = model_concordance(model, alvo_val,
                                        horizon=int(max(cfg.backtest_horizons)))
        except (ValueError, KeyError):
            c_index = None
        try:
            decile = calibration_by_decile(model, alvo_val, horizon=cfg.decile_horizon,
                                           n_bins=cfg.n_bins, alpha=cfg.alpha)
        except (ValueError, KeyError):
            decile = None
        if cfg.method == "hazard":
            try:
                ph = ph_test(des, cfg.features, baseline=cfg.baseline, link=cfg.link,
                             C=cfg.C, max_age=cfg.max_age, n_knots=cfg.n_knots,
                             alpha=cfg.alpha)
            except (ValueError, RuntimeError):
                ph = None

    return SurvivalResult(config=cfg, panel_des=des, panel_oot=oot, life_table=tab,
                          logrank=lr, model_base=base, model=model, parametric=parametric,
                          hazard_models=hazard_models, backtest=backtest,
                          discrimination=discrimination, decile=decile, c_index=c_index,
                          ph=ph, validated_on=validated_on, steps=steps)


__all__ = ["SurvivalConfig", "SurvivalResult", "run_survival_study", "split_panel",
           "STUDY_METHODS", "SPLIT_MODES", "TAILS"]
