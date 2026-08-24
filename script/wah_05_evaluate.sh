############################### WAH-NL ###############################
### SceneQueryPlanner (ours)
python -m core.run.eval --config-name=config_wah task_planner=reactstrq llm_agent.model_name='meta-llama/Meta-Llama-3.1-8B' llm_agent.em_dir='resource/trajectory/em_llm_wah/reactstrq' llm_agent.sys_prompt_path='resource/sys_prompt/wah/reactstrq.txt'

### ReAct (baseline)
#python -m core.run.eval --config-name=config_wah task_planner=react llm_agent.model_name='meta-llama/Meta-Llama-3.1-8B' llm_agent.em_dir='resource/trajectory/em_llm_wah/react' llm_agent.sys_prompt_path='resource/sys_prompt/wah/react.txt'

### ReAct + Naive Working Memory (baseline)
#python -m core.run.eval --config-name=config_wah task_planner=reactwm llm_agent.model_name='meta-llama/Meta-Llama-3.1-8B' llm_agent.em_dir='resource/trajectory/em_llm_wah/reactwm' llm_agent.sys_prompt_path='resource/sys_prompt/wah/reactwm.txt'

### SayPlan (baseline)
#python -m core.run.eval --config-name=config_wah task_planner=sayplan llm_agent.model_name='meta-llama/Meta-Llama-3.1-8B' llm_agent.em_dir='resource/trajectory/em_llm_wah/sayplan' llm_agent.sys_prompt_path='resource/sys_prompt/wah/sayplan.txt'

### MoMa-LLM (baseline)
#python -m core.run.eval --config-name=config_wah task_planner=moma llm_agent.model_name='meta-llama/Meta-Llama-3.1-8B' llm_agent.em_dir='resource/trajectory/em_llm_wah/moma' llm_agent.sys_prompt_path='resource/sys_prompt/wah/moma.txt'

