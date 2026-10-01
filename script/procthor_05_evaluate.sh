############################### AttPlan-Bench ###############################
### SceneQueryPlanner (ours)
python -m core.run.eval --config-name=config_procthor task_planner=reactstrq llm_agent.em_dir='resource/trajectory/em_llm_procthor/reactstrq' llm_agent.sys_prompt_path='resource/sys_prompt/procthor/reactstrq.txt' llm_agent.model_name='meta-llama/Llama-3.1-8B' llm_agent.max_decision_step=199

### ReAct (baseline)
#python -m core.run.eval --config-name=config_procthor task_planner=react llm_agent.em_dir='resource/trajectory/em_llm_procthor/react' llm_agent.sys_prompt_path='resource/sys_prompt/procthor/react.txt' llm_agent.model_name='meta-llama/Llama-3.1-8B' llm_agent.max_decision_step=199

### ReAct + Naive Working Memory (baseline)
#python -m core.run.eval --config-name=config_procthor task_planner=reactwm llm_agent.em_dir='resource/trajectory/em_llm_procthor/reactwm' llm_agent.sys_prompt_path='resource/sys_prompt/procthor/reactwm.txt' llm_agent.model_name='meta-llama/Llama-3.1-8B' llm_agent.max_decision_step=199

### SayPlan (baseline)
#python -m core.run.eval --config-name=config_procthor task_planner=sayplan llm_agent.em_dir='resource/trajectory/em_llm_procthor/sayplan' llm_agent.sys_prompt_path='resource/sys_prompt/procthor/sayplan.txt' llm_agent.model_name='meta-llama/Llama-3.1-8B' llm_agent.max_decision_step=199

### MoMa-LLM (baseline)
#python -m core.run.eval --config-name=config_procthor task_planner=moma llm_agent.em_dir='resource/trajectory/em_llm_procthor/moma' llm_agent.sys_prompt_path='resource/sys_prompt/procthor/moma.txt' llm_agent.model_name='meta-llama/Llama-3.1-8B' llm_agent.max_decision_step=199


############################### AttPlan-Bench (Attribute-Centric) ###############################
### SceneQueryPlanner (ours)
python -m core.run.eval --config-name=config_procthor_attribute task_planner=reactstrq llm_agent.sys_prompt_path='resource/sys_prompt/procthor/reactstrq.txt' llm_agent.model_name='meta-llama/Llama-3.1-8B' llm_agent.max_decision_step=199

### ReAct (baseline)
#python -m core.run.eval --config-name=config_procthor_attribute task_planner=react llm_agent.sys_prompt_path='resource/sys_prompt/procthor/react.txt' llm_agent.model_name='meta-llama/Llama-3.1-8B' llm_agent.max_decision_step=199

### ReAct + Naive Working Memory (baseline)
#python -m core.run.eval --config-name=config_procthor_attribute task_planner=reactwm llm_agent.sys_prompt_path='resource/sys_prompt/procthor/reactwm.txt' llm_agent.model_name='meta-llama/Llama-3.1-8B' llm_agent.max_decision_step=199

### SayPlan (baseline)
#python -m core.run.eval --config-name=config_procthor_attribute task_planner=sayplan llm_agent.sys_prompt_path='resource/sys_prompt/procthor/sayplan.txt' llm_agent.model_name='meta-llama/Llama-3.1-8B' llm_agent.max_decision_step=199

### MoMa-LLM (baseline)
#python -m core.run.eval --config-name=config_procthor_attribute task_planner=moma llm_agent.sys_prompt_path='resource/sys_prompt/procthor/moma.txt' llm_agent.model_name='meta-llama/Llama-3.1-8B' llm_agent.max_decision_step=199
