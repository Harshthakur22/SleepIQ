
plt.show()

# Drop original Sleep Quality Score column
X = df.drop(columns=["Sleep_Quality_Score", "Sleep_Quality_Label"])
y = df["Sleep_Quality_Label"]

# Train-test split