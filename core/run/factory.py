"""Builders that instantiate the environment, LLM agent, and task planner
for the simulator selected by cfg.dataset_type ('wah' or 'procthor')."""


def build_env(cfg):
    if cfg.dataset_type == 'wah':
        from core.env.wah_env import WahEnv
        return WahEnv(cfg)
    elif cfg.dataset_type == 'procthor':
        from core.env.procthor_env import ProcThorEnv
        return ProcThorEnv(cfg)
    raise ValueError(f"Unknown dataset_type: {cfg.dataset_type}")


def build_llm_agent(cfg):
    if cfg.dataset_type == 'wah':
        if cfg.task_planner == 'sayplan':
            from core.llm_agent.wah_sayplan_agent import WahSayPlanAgent
            return WahSayPlanAgent(cfg)
        elif cfg.task_planner == 'moma':
            from core.llm_agent.wah_moma_agent import WahMoMaAgent
            return WahMoMaAgent(cfg)
        else:
            from core.llm_agent.wah_reactstrq_agent import WahReActStrQAgent
            return WahReActStrQAgent(cfg)
    elif cfg.dataset_type == 'procthor':
        if cfg.task_planner == 'sayplan':
            from core.llm_agent.procthor_sayplan_agent import ProcThorSayPlanAgent
            return ProcThorSayPlanAgent(cfg)
        elif cfg.task_planner == 'moma':
            from core.llm_agent.procthor_moma_agent import ProcThorMoMaAgent
            return ProcThorMoMaAgent(cfg)
        else:
            from core.llm_agent.procthor_reactstrq_agent import ProcThorReActStrQAgent
            return ProcThorReActStrQAgent(cfg)
    raise ValueError(f"Unknown dataset_type: {cfg.dataset_type}")


def build_planner(cfg, env, llm_agent):
    if cfg.dataset_type == 'wah':
        from core.retriever.interface_wah import SceneGraphInterface_One
        from core.retriever.wah_retriever import WahRetriever
        if cfg.task_planner == 'react':
            from core.planner.wah.wah_react import WahReAct
            return WahReAct(cfg, env, llm_agent)
        elif cfg.task_planner == 'reactstrq':
            from core.planner.wah.wah_reactstrq import WahReActStrQ
            retriever = WahRetriever(cfg, None, SceneGraphInterface_One())
            return WahReActStrQ(cfg, env, llm_agent, retriever)
        elif cfg.task_planner == 'reactwm':
            from core.planner.wah.wah_reactwm import WahReActWM
            return WahReActWM(cfg, env, llm_agent)
        elif cfg.task_planner == 'sayplan':
            from core.planner.wah.wah_sayplan import WahSayPlan
            return WahSayPlan(cfg, env, llm_agent)
        elif cfg.task_planner == 'moma':
            from core.planner.wah.wah_moma import WahMoMa
            retriever = WahRetriever(cfg, None, SceneGraphInterface_One())
            return WahMoMa(cfg, env, llm_agent, retriever)
    elif cfg.dataset_type == 'procthor':
        from core.retriever.interface_procthor import SceneGraphInterface_One
        from core.retriever.procthor_retriever import ProcThorRetriever
        if cfg.task_planner == 'react':
            from core.planner.procthor.procthor_react import ProcThorReAct
            return ProcThorReAct(cfg, env, llm_agent)
        elif cfg.task_planner == 'reactstrq':
            from core.planner.procthor.procthor_reactstrq import ProcThorReActStrQ
            retriever = ProcThorRetriever(cfg, None, SceneGraphInterface_One())
            return ProcThorReActStrQ(cfg, env, llm_agent, retriever)
        elif cfg.task_planner == 'reactwm':
            from core.planner.procthor.procthor_reactwm import ProcThorReActWM
            return ProcThorReActWM(cfg, env, llm_agent)
        elif cfg.task_planner == 'sayplan':
            from core.planner.procthor.procthor_sayplan import ProcThorSayPlan
            return ProcThorSayPlan(cfg, env, llm_agent)
        elif cfg.task_planner == 'moma':
            from core.planner.procthor.procthor_moma import ProcThorMoMa
            retriever = ProcThorRetriever(cfg, None, SceneGraphInterface_One())
            return ProcThorMoMa(cfg, env, llm_agent, retriever)
    raise ValueError(f"Unknown task_planner '{cfg.task_planner}' for dataset_type '{cfg.dataset_type}'")
