import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

developers=np.array([2,4,5,6,7,8]).reshape(-1,1)
days=np.array([76,43,40,36,39,25])

model=LinearRegression()

model.fit(developers,days)

predicted_data =model.predict(np.array([[6]]))
print(f"completed work by 6 developer  {int(predicted_data[0])} days")
plt.scatter(developers,days,color='red')
plt.plot(developers,model.predict(developers),  color='blue')
plt.xlabel('No. of Developers')
plt.show()
