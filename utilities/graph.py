import matplotlib.pyplot as plt
import seaborn as sns

def convert_snake_case(label, capitalize_first=True, capitalize_all=False):
# Code directly taken from breakfast hour lecture: https://git.generalassemb.ly/DSIR-222/breakfast-hour/tree/master/06_week/reusable-graphing-fx
    """Takes an input string assumed to be in snake case and returns it with spaces inserted and capitalized to your liking.

    Keyword Arguments:
    label: a_snake_case_string
    capitalize_first: Bool, capitalize the first word of the string
    capitalize_all: Bool, capitalize all words of the string"""

    if capitalize_all:
        return ' '.join([word.capitalize() for word in label.split('_')])
    elif capitalize_first:
        return ' '.join([word.capitalize() if i == 0 else word for i, word in enumerate(label.split('_'))])
    else:
        return ' '.join([word for word in label.split('_')])

def my_barplot(df, x_col, y_col, title, size):
    # Set figure size
    plt.figure(figsize=(18,6))
    
    # Create plot
    sns.barplot(x=df[x_col], y=df[y_col]);
    
    # Set title
    plt.title(title, fontsize=size)
    
    # Set labels and ticks
    plt.xlabel(convert_snake_case(label=x_col, capitalize_all=True), fontsize = size - 4, labelpad = size/2)
    plt.xticks(fontsize=size/2, rotation=45)
    plt.ylabel(convert_snake_case(label=y_col, capitalize_all=True), fontsize = size - 4, labelpad = size/2)
    plt.yticks(fontsize=size/2);
    