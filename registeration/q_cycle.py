from sympy import FiniteField


def generate_cyclic_group(q):
    # Create a finite field of order q
    F_q = FiniteField(q)

    # Find a generator for the cyclic group
    generator = None
    for i in range(1, q):
        # Check if i is a generator
        if F_q(i) ** (q - 1) != 1:
            generator = F_q(i)
            break

    if generator is None:
        raise ValueError("No generator found for the cyclic group.")

    # Generate the cyclic group using the generator
    cyclic_group_elements = [generator ** i for i in range(q)]

    return F_q, generator, cyclic_group_elements


# Example usage:
q = 100  # Choose the order of the finite field and cyclic group
F_q, generator, cyclic_group_elements = generate_cyclic_group(q)

print("Finite Field F_{}: {}".format(q, F_q))
print("Generator of the cyclic group: {}".format(generator))
print("Cyclic Group Elements over F_{}: {}".format(q, cyclic_group_elements))
