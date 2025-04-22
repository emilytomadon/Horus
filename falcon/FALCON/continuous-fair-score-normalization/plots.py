import numpy as np
import matplotlib.pyplot as plt
import pickle

#fnmr = frr -> how often we reject a person who should be accepted
#fmr = far -> how often we accept a person who should be rejected
#eer -> the point where these two lines intercept

MODELS_PATH = '/home/gimicaroni/Documents/Unicamp/IC/HorusProjeto/HorusEthnicity/Horus/fsn/fair_score_normalization/data/models'

if __name__ == "__main__":
    overall = np.array([0.007550335570469802, 0.007550335570469802])*100
    african = np.array([0.018153526970954403, 0.01919087136929465])*100
    asian = np.array([0.01639811778126332, 0.01768144873805788])*100
    caucasian = np.array([0.011842327863305746, 0.011673151750972721])*100
    indian = np.array([0.02140622427136507, 0.021241561007739218])*100
    diff_overall = (overall[1] - overall[0])*100/(100-overall[0])
    diff_african = (african[1] - african[0])*100/(100-african[0])
    diff_asian = (asian[1] - asian[0])*100/(100-asian[0])
    diff_caucasian = (caucasian[1] - caucasian[0])*100/(100-caucasian[0])
    diff_indian = (indian[1] - indian[0])*100/(100-indian[0])
    results = {0: diff_overall, 1: diff_african, 2: diff_asian, 3: diff_caucasian, 4: diff_indian}
    print(100-overall[0], 100-african[0], 100-asian[0], 100-caucasian[0], 100-indian[0])
    print(100-overall[1], 100-african[1], 100-asian[1], 100-caucasian[1], 100-indian[1])

    # Compute differences

    # # Plot only the improvements
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color="blue")  # Convert to percentage
    plt.ylabel("Accuracy Change (%)")
    plt.title("Effect after Normalization (FALCON)")
    plt.xticks([0, 1, 2, 3, 4], ['Overall', 'African', 'Asian', 'Caucasian', 'Indian'])

    # # Add value labels
    # for i, v in enumerate(diff):
    #     plt.text(clusters[i], v - 5, f"{v:.3f}%", ha='center')



    # # for i, (a1, a2) in enumerate(zip(acc_unnormalized, acc_normalized)):
    # #     plt.text(i - bar_width/2, a1, f"{a1:.2f}%", ha='center', fontsize=10)
    # #     plt.text(i + bar_width/2, a2, f"{a2:.2f}%", ha='center', fontsize=10)

    # # Show the plot
    plt.show()


