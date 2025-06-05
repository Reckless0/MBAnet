# Dealing with continuous variables
def Normalize(df):
    means,stds = {},{}

    cont_names = ['TB', 'DB', 'ALT', 'AST', 'GGT', 'TBA', 'ALB(g/L)', 'PLT', 'AGE']
    for n in cont_names:
        # assert pd.is_numeric_dtype(df[n]), (f"""Cannot normalize '{n}' column as it isn't numerical. Are you sure it doesn't belong in the categorical set of columns?""")
        means[n],stds[n] = df[n].mean(),df[n].std()
        df[n] = (df[n]-means[n]) / (1e-7 + stds[n])

    return df

# Dealing with continuous variables
def Normalize_test(df):
    means,stds = {},{}

    cont_names = ['TB', 'DB', 'ALT', 'AST', 'GGT', 'AGE']
    for n in cont_names:
        # assert pd.is_numeric_dtype(df[n]), (f"""Cannot normalize '{n}' column as it isn't numerical. Are you sure it doesn't belong in the categorical set of columns?""")
        means[n],stds[n] = df[n].mean(),df[n].std()
        df[n] = (df[n]-means[n]) / (1e-7 + stds[n])

    return df
