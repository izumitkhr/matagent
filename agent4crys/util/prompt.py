def get_prompt(target_value, target_prop="formation_energy_per_atom"):

    if target_prop == "formation_energy_per_atom":
        unit = "eV/atom"
    else:
        raise ValueError(f"Unknown target property: {target_prop}")

    prompt = f"I am looking to design a material with a formation energy per atom of {target_value} {unit}."
    return prompt
