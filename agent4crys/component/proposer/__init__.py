from .gpt import OAProposer
from .tf import TFProposer
from .planner import Planner


def load_proposer(args, target_prompt, knowledge_base, max_new_tokens_for_tf_proposer=2048, device="cuda"):
    llm_model = args.llm_model
    if llm_model == "gpt-4o":
        model_id = "gpt-4o-2024-08-06"
        proposer = OAProposer(
            target_val=args.target_value,
            target_prompt=target_prompt,
            knowledge_base=knowledge_base,
            gpt_model=model_id,
        )
        if args.use_planning:
            return Planner(proposer)
        else:
            return proposer
    elif llm_model == "o1":
        model_id = "o1-2024-12-17"
        proposer = OAProposer(
            target_val=args.target_value,
            target_prompt=target_prompt,
            knowledge_base=knowledge_base,
            gpt_model=model_id,
        )
        if args.use_planning:
            return Planner(proposer)
        else:
            return proposer
    elif llm_model == "o3-mini":
        model_id = "o3-mini-2025-01-31"
        proposer = OAProposer(
            target_val=args.target_value,
            target_prompt=target_prompt,
            knowledge_base=knowledge_base,
            gpt_model=model_id,
        )
        if args.use_planning:
            return Planner(proposer)
        else:
            return proposer
    elif llm_model == "gpt-3.5-turbo":
        model_id = "gpt-3.5-turbo-0125"
        proposer = OAProposer(
            target_val=args.target_value,
            target_prompt=target_prompt,
            knowledge_base=knowledge_base,
            gpt_model=model_id,
        )
        if args.use_planning:
            return Planner(proposer)
        else:
            return proposer
    elif llm_model in [
        "meta-llama/Llama-3.1-8B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "openai/gpt-oss-20b",
    ]:
        model_id = llm_model
        proposer = TFProposer(
            target_val=args.target_value,
            target_prompt=target_prompt,
            knowledge_base=knowledge_base,
            model_id=model_id,
            device=device,
            max_new_tokens=max_new_tokens_for_tf_proposer,
        )
        if args.use_planning:
            return Planner(proposer)
        else:
            return proposer
    else:
        raise ValueError(f"Model {llm_model} not supported")
