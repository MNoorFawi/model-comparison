library(kernlab)
library(e1071)
library(ggplot2)
library(ROCR)
library(reshape)

data('spirals')
s <- specc(spirals, centers = 2)
spiral_data <- data.frame(x = spirals[,1], y = spirals[,2],
                class = as.factor(s))

ggplot(data = spiral_data) +
  geom_point(aes(x = x, y = y, color = class)) +
  coord_fixed() + theme_bw() + 
  scale_color_brewer(palette = 'Set1')
  
  accuracy <- function(model, outcome){
  mean(model == outcome)
}
auc <- function(model, outcome) { 
  per <- performance(prediction(model, outcome == 2),
                      "auc")
  as.numeric(per@y.values)
}

# Logistic Regression
logit <- glm(class ~ x + y,
                 family = binomial(link = 'logit'),
                 data = spiral_data)
spiral_data$logit_pred <- ifelse(predict(logit) > 0, 2, 1)
accuracy(spiral_data$logit_pred, spiral_data$class)
auc(spiral_data$logit_pred, spiral_data$class)

# Naive Bayes
library(e1071)
nb <- naiveBayes(class ~ x + y, data = spiral_data)
spiral_data$nb_pred <- ifelse(
  predict(nb, newdata = spiral_data) == 2, 2, 1)
accuracy(spiral_data$nb_pred, spiral_data$class)
auc(spiral_data$nb_pred, spiral_data$class)

# SVM
svm_fit <- svm(class ~ x + y, data = spiral_data)
spiral_data$svm_pred <- ifelse(predict(svm_fit) == 2, 2, 1)
accuracy(spiral_data$svm_pred, spiral_data$class)
auc(spiral_data$svm_pred, spiral_data$class)

### ON DATA 
predictions <- melt(spiral_data, id.vars = c('x', 'y'))
ggplot(predictions, aes(x = x, y = y, color = factor(value))) +
  geom_point() + coord_fixed() +
  scale_colour_brewer(palette = 'Set1') +
  facet_wrap(~ variable) + theme_bw()
  
### decision tree
library(rpart)
tree <- rpart(class ~ x + y, data = spiral_data)
prediction <- predict(tree, newdata = spiral_data, type = 'class')

### knn
library(class)
knn15 <- knn(train = spiral_data[, 1:2], test = spiral_data[, 1:2], 
             spiral_data$class, k = 15)

spiral_data <- spiral_data[, 1:3]
# radial
radial <- svm(class ~ x + y, data = spiral_data, kernel = 'radial')
spiral_data$svm_radial <- ifelse(predict(radial) == 2, 2, 1)
# linear
linear <- svm(class ~ x + y, data = spiral_data, kernel = 'linear')
spiral_data$svm_linear <- ifelse(predict(linear) == 2, 2, 1)
# polynomial
polynomial <- svm(class ~ x + y,
                  data = spiral_data, kernel = 'polynomial')
spiral_data$svm_polynomial <- ifelse(predict(polynomial) == 2, 2, 1)
# sigmoid
sigmoid <- svm(class ~ x + y, data = spiral_data, kernel = 'sigmoid')
spiral_data$svm_sigmoid <- ifelse(predict(sigmoid) == 2, 2, 1)

for(i in 4:7){
  print(names(spiral_data)[i])
  print(accuracy(spiral_data[, i], spiral_data$class))
}

### plotting them
predictions <- melt(spiral_data, id.vars = c('x', 'y'))
ggplot(predictions, aes(x = x, y = y, color = factor(value))) +
  geom_point() + coord_fixed() +
  scale_colour_brewer(palette = 'Set1') +
  facet_wrap(~ variable) + theme_bw()

tuned_svm_radial <- tune.svm(class ~ x + y, data = spiral_data,
                      kernel = "radial", 
                  gamma = seq(0.2, 2, 0.2), 
                  cost = c(2 ^ (2:9), 10 ^ (-3:1)))
tuned_svm_radial$best.parameters

spiral_data <- spiral_data[, 1:3]

tuned_model <- svm(class ~ x + y, data = spiral_data,
                   kernel = 'radial', 
                   gamma = tuned_svm_radial$best.parameters[1],
                   cost = tuned_svm_radial$best.parameters[2])
spiral_data$tuned_svm <- ifelse(predict(tuned_model) == 2, 2, 1)

predictions <- melt(spiral_data, id.vars = c('x', 'y'))
ggplot(predictions, aes(x = x, y = y, color = factor(value))) +
  geom_point() + coord_fixed() +
  scale_colour_brewer(palette = 'Set1') +
  facet_wrap(~ variable) + theme_bw()

accuracy(spiral_data$tuned_svm, spiral_data$class)
auc(spiral_data$tuned_svm, spiral_data$class)

