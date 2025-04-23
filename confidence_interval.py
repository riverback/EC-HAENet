# Clopper–Pearson interval for accuracy, PPV, NPV, sensitivity, specificity, F1 score

from scipy.stats import beta

def generate_CI(confusion_matrix, alpha=0.05):
    alpha = alpha
    return_str = ''
    def clopper_pearson_interval(number_of_successes, number_of_trials, alpha):
        p_u, p_o = beta.ppf([alpha/2, 1-alpha/2], 
            [number_of_successes, number_of_successes + 1],
            [number_of_trials - number_of_successes + 1, number_of_trials - number_of_successes])

        #print(f'{number_of_successes / number_of_trials:.4f}', f'({p_u:.4f}-{p_o:.4f})')
        return f'{number_of_successes / number_of_trials:.4f} ({p_u:.4f}-{p_o:.4f})\n'
        
    # accuracy
    n_of_successes = confusion_matrix[0][0] + confusion_matrix[1][1]
    n_of_trials = sum(sum(row) for row in confusion_matrix)
    #print('accuracy', end=' ')
    #_ = clopper_pearson_interval(n_of_successes, n_of_trials, alpha)
    return_str += f'accuracy {clopper_pearson_interval(n_of_successes, n_of_trials, alpha)}'

    # PPV(positive predictive value)
    n_of_successes = confusion_matrix[1][1]
    n_of_trials = confusion_matrix[1][1] + confusion_matrix[0][1]
    #print('PPV', end=' ')
    #_ = clopper_pearson_interval(n_of_successes, n_of_trials, alpha)
    return_str += f'PPV {clopper_pearson_interval(n_of_successes, n_of_trials, alpha)}'

    # NPV(negative predictive value)
    n_of_successes = confusion_matrix[0][0]
    n_of_trials = confusion_matrix[0][0] + confusion_matrix[1][0]
    #print('NPV', end=' ')
    #_ = clopper_pearson_interval(n_of_successes, n_of_trials, alpha)
    return_str += f'NPV {clopper_pearson_interval(n_of_successes, n_of_trials, alpha)}'

    # sensitivity
    n_of_successes = confusion_matrix[0][0]
    n_of_trials = confusion_matrix[0][0] + confusion_matrix[0][1]
    #print('sensitivity', end=' ')
    #_ = clopper_pearson_interval(n_of_successes, n_of_trials, alpha)
    return_str += f'sensitivity {clopper_pearson_interval(n_of_successes, n_of_trials, alpha)}'

    # specificity
    n_of_successes = confusion_matrix[1][1]
    n_of_trials = confusion_matrix[1][1] + confusion_matrix[1][0]
    #print('specificity', end=' ')
    #_ = clopper_pearson_interval(n_of_successes, n_of_trials, alpha)
    return_str += f'specificity {clopper_pearson_interval(n_of_successes, n_of_trials, alpha)}'
    
    return return_str

if __name__ == '__main__':
    confusion_matrix = [
        [861, 47],
        [69, 781]]
    result = generate_CI(confusion_matrix)
    print(result)