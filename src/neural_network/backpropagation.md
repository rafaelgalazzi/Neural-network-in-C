BACKPROPAGATION FROM SCRATCH

1. Forward pass

---

output = (w1 _ x1) + (w2 _ x2) + ... + bias

2. Calculate loss

---

error = output - expected
loss = 0.5 \* error^2

3. Derivative of loss

---

dLoss_dOutput = output - expected

4. Gradient for each weight

---

dLoss_dWeight_i = dLoss_dOutput \* input_i

Example:

dLoss_dw1 = dLoss_dOutput _ x1
dLoss_dw2 = dLoss_dOutput _ x2

5. Gradient for bias

---

dLoss_dBias = dLoss_dOutput

6. Update weights

---

weight_i = weight_i - learningRate \* dLoss_dWeight_i

Example:

w1 -= lr _ dLoss_dw1
w2 -= lr _ dLoss_dw2

7. Update bias

---

bias -= lr \* dLoss_dBias

8. Repeat

---

Run this for all training samples and repeat for many epochs.
