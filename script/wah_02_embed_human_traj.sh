############################### WAH-NL ###############################
### SceneQueryPlanner (ours)
python -m core.run.embed_em_wah --config-name=config_wah task_planner=reactstrq dataset.collect_dir='resource/trajectory/collect_human_wah/reactstrq' llm_agent.em_dir='resource/trajectory/em_human_wah/reactstrq'

### ReAct (baseline)
#python -m core.run.embed_em_wah --config-name=config_wah task_planner=react dataset.collect_dir='resource/trajectory/collect_human_wah/react' llm_agent.em_dir='resource/trajectory/em_human_wah/react'

### ReAct + Naive Working Memory (baseline)
#python -m core.run.embed_em_wah --config-name=config_wah task_planner=reactwm dataset.collect_dir='resource/trajectory/collect_human_wah/reactwm' llm_agent.em_dir='resource/trajectory/em_human_wah/reactwm'

### SayPlan (baseline)
#python -m core.run.embed_em_sayplan_wah --config-name=config_wah task_planner=sayplan dataset.collect_dir='resource/trajectory/collect_human_wah/sayplan' llm_agent.em_dir='resource/trajectory/em_human_wah/sayplan'

### MoMa-LLM (baseline)
#python -m core.run.embed_em_moma_wah --config-name=config_wah task_planner=moma dataset.collect_dir='resource/trajectory/collect_human_wah/moma' llm_agent.em_dir='resource/trajectory/em_human_wah/moma'

