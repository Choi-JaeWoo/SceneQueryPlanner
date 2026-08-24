############################### AttPlan-Bench ###############################
### SceneQueryPlanner (ours)
python -m core.run.collect_human --config-name=config_procthor task_planner=reactstrq dataset.collect_dir='resource/trajectory/collect_human_procthor/reactstrq' procthor.eval_set='dataset/InContext_Example.json' llm_agent.em_dir='resource/trajectory/em_human_procthor/reactstrq' llm_agent.sys_prompt_path='resource/sys_prompt/procthor/reactstrq.txt'

### ReAct (baseline)
#python -m core.run.collect_human --config-name=config_procthor task_planner=react dataset.collect_dir='resource/trajectory/collect_human_procthor/react' procthor.eval_set='dataset/InContext_Example.json' llm_agent.em_dir='resource/trajectory/em_human_procthor/react' llm_agent.sys_prompt_path='resource/sys_prompt/procthor/react.txt'

### ReAct + Naive Working Memory (baseline)
#python -m core.run.collect_human --config-name=config_procthor task_planner=reactwm dataset.collect_dir='resource/trajectory/collect_human_procthor/reactwm' procthor.eval_set='dataset/InContext_Example.json' llm_agent.em_dir='resource/trajectory/em_human_procthor/reactwm' llm_agent.sys_prompt_path='resource/sys_prompt/procthor/reactwm.txt'

### SayPlan (baseline)
#python -m core.run.collect_human --config-name=config_procthor task_planner=sayplan dataset.collect_dir='resource/trajectory/collect_human_procthor/sayplan' procthor.eval_set='dataset/InContext_Example.json' llm_agent.em_dir='resource/trajectory/em_human_procthor/sayplan' llm_agent.sys_prompt_path='resource/sys_prompt/procthor/sayplan.txt'

### MoMa-LLM (baseline)
#python -m core.run.collect_human --config-name=config_procthor task_planner=moma dataset.collect_dir='resource/trajectory/collect_human_procthor/moma' procthor.eval_set='dataset/InContext_Example.json' llm_agent.em_dir='resource/trajectory/em_human_procthor/moma' llm_agent.sys_prompt_path='resource/sys_prompt/procthor/moma.txt'


############################### AttPlan-Bench (Attribute-Centric) ###############################
### SceneQueryPlanner (ours)
python -m core.run.collect_human --config-name=config_procthor task_planner=reactstrq dataset.collect_dir='resource/trajectory/collect_human_procthor/using_attributes/reactstrq' procthor.eval_set='dataset/InContext_UsingAttributesTask.json' llm_agent.em_dir='resource/trajectory/em_human_procthor/using_attributes/reactstrq' llm_agent.sys_prompt_path='resource/sys_prompt/procthor/reactstrq.txt'

### ReAct (baseline)
#python -m core.run.collect_human --config-name=config_procthor task_planner=react dataset.collect_dir='resource/trajectory/collect_human_procthor/using_attributes/react' procthor.eval_set='dataset/InContext_UsingAttributesTask.json' llm_agent.em_dir='resource/trajectory/em_human_procthor/using_attributes/react' llm_agent.sys_prompt_path='resource/sys_prompt/procthor/react.txt'

### ReAct + Naive Working Memory (baseline)
#python -m core.run.collect_human --config-name=config_procthor task_planner=reactwm dataset.collect_dir='resource/trajectory/collect_human_procthor/using_attributes/reactwm' procthor.eval_set='dataset/InContext_UsingAttributesTask.json' llm_agent.em_dir='resource/trajectory/em_human_procthor/using_attributes/reactwm' llm_agent.sys_prompt_path='resource/sys_prompt/procthor/reactwm.txt'

### SayPlan (baseline)
#python -m core.run.collect_human --config-name=config_procthor task_planner=sayplan dataset.collect_dir='resource/trajectory/collect_human_procthor/using_attributes/sayplan' procthor.eval_set='dataset/InContext_UsingAttributesTask.json' llm_agent.em_dir='resource/trajectory/em_human_procthor/using_attributes/sayplan' llm_agent.sys_prompt_path='resource/sys_prompt/procthor/sayplan.txt'

### MoMa-LLM (baseline)
#python -m core.run.collect_human --config-name=config_procthor task_planner=moma dataset.collect_dir='resource/trajectory/collect_human_procthor/using_attributes/moma' procthor.eval_set='dataset/InContext_UsingAttributesTask.json' llm_agent.em_dir='resource/trajectory/em_human_procthor/using_attributes/moma' llm_agent.sys_prompt_path='resource/sys_prompt/procthor/moma.txt'