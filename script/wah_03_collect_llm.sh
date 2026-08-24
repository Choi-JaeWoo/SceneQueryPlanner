############################### WAH-NL ###############################
### SceneQueryPlanner (ours)
python -m core.run.collect_llm --config-name=config_wah task_planner=reactstrq dataset.collect_dir='resource/trajectory/collect_llm_wah/reactstrq' llm_agent.em_dir='resource/trajectory/em_human_wah/reactstrq' llm_agent.sys_prompt_path='resource/sys_prompt/wah/reactstrq.txt'

### ReAct (baseline)
#python -m core.run.collect_llm --config-name=config_wah task_planner=react dataset.collect_dir='resource/trajectory/collect_llm_wah/react' llm_agent.em_dir='resource/trajectory/em_human_wah/react' llm_agent.sys_prompt_path='resource/sys_prompt/wah/react.txt'

### ReAct + Naive Working Memory (baseline)
#python -m core.run.collect_llm --config-name=config_wah task_planner=reactwm dataset.collect_dir='resource/trajectory/collect_llm_wah/reactwm' llm_agent.em_dir='resource/trajectory/em_human_wah/reactwm' llm_agent.sys_prompt_path='resource/sys_prompt/wah/reactwm.txt'

### SayPlan (baseline)
#python -m core.run.collect_llm --config-name=config_wah task_planner=sayplan dataset.collect_dir='resource/trajectory/collect_llm_wah/sayplan' llm_agent.em_dir='resource/trajectory/em_human_wah/sayplan' llm_agent.sys_prompt_path='resource/sys_prompt/wah/sayplan.txt'

### MoMa-LLM (baseline)
#python -m core.run.collect_llm --config-name=config_wah task_planner=moma dataset.collect_dir='resource/trajectory/collect_llm_wah/moma' llm_agent.em_dir='resource/trajectory/em_human_wah/moma' llm_agent.sys_prompt_path='resource/sys_prompt/wah/moma.txt'

