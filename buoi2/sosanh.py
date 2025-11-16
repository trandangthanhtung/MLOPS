import pandas as pd
import matplotlib.pyplot as plt
df_no = pd.read_csv('outputs/results_no_preprocess.csv')
df_p = pd.read_csv('outputs/results_preprocess.csv')


df_comp = pd.DataFrame({
    'Model': df_no['model'],
    'Accuracy_NoPreprocess': df_no['accuracy'],
    'F1_NoPreprocess': df_no['f1'],
    'Accuracy_Preprocess': df_p['accuracy'],
    'F1_Preprocess': df_p['f1']
})

print(df_comp)


df_comp.set_index('Model')[['F1_NoPreprocess','F1_Preprocess']].plot(kind='bar', figsize=(10,6))
plt.title('So sánh F1 score: Không tiền xử lý vs Có tiền xử lý')
plt.ylabel('F1 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/f1_comparison.png')
plt.show()