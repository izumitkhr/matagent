import argparse
import warnings

import torch
from tqdm import tqdm

from agent4crys.util import set_seed, get_prompt
from agent4crys.diffusion.pretrain import load_diffusion_model, run_diffusion_model
from agent4crys.component import (
    load_evaluator,
    load_memory,
    load_knowledge_base,
    load_proposer,
)
from agent4crys.component.proposer.util import get_initial_guess

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for matagent")

    parser.add_argument(
        "--initial_guess",
        type=str,
        choices=["random", "llm", "retriever"],
        default="random",
        help="Initial guess for the optimization",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/mp_20/train.csv",
        help="Path to the data file",
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        choices=[
            "o1",
            "o3-mini",
            "gpt-4o",
            # "gpt-3.5-turbo",
            # "claude-3-5",
        ],
        default="gpt-4o",
        help="LLM to use",
    )

    parser.add_argument(
        "--target_value",
        type=float,
        default=-3.8,
        help="Target value for the optimization",
    )

    parser.add_argument(
        "--n_init",
        type=int,
        default=5,
        help="Number of initial guesses to generate",
    )

    parser.add_argument(
        "--n_iterations",
        type=int,
        default=20,
        help="Max number of iterations for one initial guess",
    )

    parser.add_argument(
        "--chars",
        type=int,
        default=300,
        help="Number of characters for justification statement",
    )

    parser.add_argument(
        "--use_short_term_memory",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--max_short_term_memory_size",
        type=int,
        default=3,
        help="Maximum size of short-term memory",
    )

    parser.add_argument(
        "--use_long_term_memory",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--max_long_term_memory_size",
        type=int,
        default=3,
        help="Maximum size of long-term memory",
    )

    parser.add_argument(
        "--use_kb_table",
        action="store_true",
        default=False,
        help="Use knowledge base for generation",
    )

    parser.add_argument(
        "--use_kb_reason",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--use_planning",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--additional_prompt",
        type=str,
        default="",
    )

    parser.add_argument(
        "--set_seed",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument("--no_cuda", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Device: {device}")
    print(f"Planning: {args.use_planning}")
    if args.set_seed:
        set_seed(args.seed)
    prompt = get_prompt(args.target_value)
    print("Setting up models ...")
    # load diffusion models
    print("Loading diffusion models ...")
    diffusion_model = load_diffusion_model().to(device)
    diffusion_model.device = device
    # set up evaluator
    print("Loading evaluation models ...")
    evaluator = load_evaluator(args, device)
    # set up knowledge base
    print("Setting up knowledge bases ...")
    knowledge_base = load_knowledge_base()
    # set up llm model
    print("Loading LLM models ...")
    proposer = load_proposer(
        args=args, target_prompt=prompt, knowledge_base=knowledge_base
    )
    # set initial guess
    if args.initial_guess == "retriever":
        initial_guesses = get_initial_guess(
            args,
            proposer,
            prompt,
            additional_prompt=args.additional_prompt,
            device=device,
        )

    # ==================== Inference ====================

    print("Start inference.")
    for i in range(args.n_init):
        memory = load_memory(args)
        prev_valid = True
        with open(f"response_log_{i:02d}.txt", "w") as file:
            iter = 0
            # get initial guess
            if args.initial_guess == "retriever":
                next_guess = initial_guesses[i]
            else:
                next_guess = get_initial_guess(args, proposer, prompt, device=device)
            file.write("##################################################\n")
            file.write(f" Iteration {iter} (Initialization)\n")
            file.write("##################################################\n")
            file.write(str(next_guess) + "\n")
            # diffusion
            gen_mat = run_diffusion_model(next_guess, diffusion_model)
            gen_mat_df = evaluator.evaluate(gen_mat)
            gen_mat_df.to_csv(f"gen_mat_init{i:02d}_iter{iter:02d}.csv")
            feedback, predicted_val = evaluator.feedback(gen_mat_df)
            file.write(f"Feedback: {feedback}\n")
            memory.store(next_guess, feedback, predicted_val)
            for iter in range(1, args.n_iterations + 1):
                file.write("##################################################\n")
                file.write(f" iteration {iter}\n")
                file.write("##################################################\n")
                if args.use_planning:
                    next_guess = proposer.plan_and_execute(
                        memory=memory,
                        prev_guess=next_guess,
                        feedback=feedback,
                        prev_valid=prev_valid,
                        chars=args.chars,
                        file=file,
                        additional_prompt=args.additional_prompt,
                    )
                elif args.use_short_term_memory:
                    next_guess = proposer.retrieve_and_propose_short_term(
                        memory=memory,
                        prev_valid=prev_valid,
                        prev_guess=next_guess,
                        feedback=feedback,
                        file=file,
                        chars=args.chars,
                        additional_prompt=args.additional_prompt,
                    )
                elif args.use_long_term_memory:
                    next_guess = proposer.retrieve_and_propose_long_term(
                        memory=memory,
                        prev_valid=prev_valid,
                        prev_guess=next_guess,
                        feedback=feedback,
                        file=file,
                        chars=args.chars,
                        additional_prompt=args.additional_prompt,
                    )
                elif args.use_kb_table:
                    next_guess = proposer.retrieve_and_propose_table(
                        memory=memory,
                        prev_valid=prev_valid,
                        prev_guess=next_guess,
                        feedback=feedback,
                        file=file,
                        chars=args.chars,
                        additional_prompt=args.additional_prompt,
                    )
                elif args.use_kb_reason:
                    next_guess = proposer.retrieve_and_propose_reason(
                        memory=memory,
                        prev_valid=prev_valid,
                        prev_guess=next_guess,
                        feedback=feedback,
                        file=file,
                        chars=args.chars,
                        additional_prompt=args.additional_prompt,
                    )
                else:
                    next_guess = proposer.propose_one(
                        memory=memory,
                        prev_guess=next_guess,
                        feedback=feedback,
                        file=file,
                        additional_prompt=args.additional_prompt,
                    )

                # check if next guess is valid
                prev_valid, feedback = evaluator.check_validity(next_guess)
                if prev_valid:
                    pass
                else:
                    if args.use_planning:
                        memory.delete_plan()
                    file.write("# Feedback:\n")
                    file.write(f"{feedback}\n")
                    continue

                gen_mat_df = run_diffusion_model(next_guess, diffusion_model)
                gen_mat_df = evaluator.evaluate(gen_mat_df)
                gen_mat_df.to_csv(f"gen_mat_init{i:02d}_iter{iter:02d}.csv")
                feedback, predicted_val = evaluator.feedback(gen_mat_df)
                file.write("# Feedback:\n")
                file.write(f"{feedback}\n")
                memory.store(next_guess, feedback, predicted_val)
                print(f"Iteration {iter} done.")
