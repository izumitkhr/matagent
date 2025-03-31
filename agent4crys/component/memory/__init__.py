from .memory import Memory


def load_memory(args):
    return Memory(
        max_short_term_size=args.max_short_term_memory_size,
        max_long_term_size=args.max_long_term_memory_size,
        use_short_term_memory=args.use_short_term_memory,
        use_long_term_memory=args.use_long_term_memory,
    )
