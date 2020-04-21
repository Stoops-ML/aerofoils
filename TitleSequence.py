def print_title(strings, spacing=60):
    print("#"*spacing)
    print(f"#{'Aerofoil regression problem':^{spacing-2}}#")
    for string in strings:
        print(f"# {string:<{spacing-4}} #")
    print("#"*spacing)
    print()
