knitr::opts_chunk$set(echo = TRUE)

# Libraries
library(ggplot2)
library(dplyr)
library(corrplot)
library(gridExtra)
library(caret)
library(car)
library(ROCR)
library(pROC)
library(ggeffects)
library(MASS)


# ----

bank <- read.csv("C:/Users/Michele Puglia/Desktop/bank.csv", header = TRUE)
glimpse(bank)
summary(bank)


# ----

bank <- bank %>%
  mutate(
    churn = as.factor(churn),
    country = as.factor(country),  #(1=francia, 2=germania, 3=spagna)
    gender = as.factor(gender),
    active_member = as.factor(active_member),
    credit_card = as.factor(credit_card)
  )


# ----

anyNA(bank)


# ----

#Churn
bank %>%
  count(churn) %>%
  mutate(proportion = n / sum(n)) %>%
  rename(churn = churn, proportion = proportion) %>%
  print()

ggplot(bank, aes(x = churn, fill = churn)) +
  geom_bar() +
  labs(title = "Distribuzione di 'churn'",
       x = "Churn",
       y = "Conteggio") +
  scale_fill_manual(values = c("steelblue", "orange"))+
  theme_classic()


# ----

#Gender e country rispetto a churn
bar_gender_churn<-ggplot(bank, aes(x = gender, fill = churn)) +
  geom_bar(position = "stack", stat = "count") +  #position=stack permette di creare delle barre impilate
  labs(title = "Distribuzione Churn per Gender",
       x = "Gender",
       y = "Conteggio") +
  scale_fill_manual(values = c("steelblue", "orange"),
                    name = "Churn",
                    labels = c("No Churn", "Churn"))

bar_country_churn<-ggplot(bank, aes(x = country, fill = churn)) +
  geom_bar(position = "stack", stat = "count") +
  labs(title = "Distribuzione Churn-Country",
       x = "Country",
       y = "Conteggio") +
  scale_fill_manual(values = c("steelblue", "orange"),
                    name = "Churn",
                    labels = c("No Churn", "Churn"))

grid.arrange(bar_gender_churn, bar_country_churn, ncol = 2)


# ----


bar_active_churn<-ggplot(bank, aes(x = active_member, fill = churn)) +
  geom_bar(position = "stack", stat = "count") +
  labs(title = "Distribuzione Churn-Active_member",
       x = "Active_member",
       y = "Conteggio") +
  scale_fill_manual(values = c("steelblue", "orange"),
                    name = "Churn",
                    labels = c("No Churn", "Churn"))

bar_creditcard_churn<-ggplot(bank, aes(x =credit_card, fill = churn)) +
  geom_bar(position = "stack", stat = "count") +
  labs(title = "Distribuzione Churn-credit_card",
       x = "Credit_card",
       y = "Conteggio") +
  scale_fill_manual(values = c("steelblue", "orange"),
                    name = "Churn",
                    labels = c("No Churn", "Churn"))

grid.arrange(bar_active_churn, bar_creditcard_churn, ncol = 2)


# ----

#Age
hist_age<-ggplot(bank, aes(x = age, fill = churn)) +
  geom_histogram(binwidth = 1, color = "black", position = "identity", alpha = 0.7) +
  scale_fill_manual(values = c("steelblue", "orange")) +
  labs(title = "Distribuzione di Age") +
  xlab("Age") +
  ylab("Frequenza")

bp_age<-ggplot(bank, aes(x = "", y = age)) +
  geom_boxplot(fill="steelblue") +
  labs(title = "Boxplot di Age",
       x = "",
       y = "Age")

grid.arrange(hist_age, bp_age, ncol = 2)



# ----

#Balance
hist_balance<-ggplot(bank, aes(x = balance, fill = churn)) +
  geom_histogram(binwidth = 10000, color = "black", position = "identity", alpha = 0.7) +
  scale_fill_manual(values = c("steelblue", "orange")) +
  labs(title = "Distribuzione di balance") +
  xlab("balance") +
  ylab("Frequenza") +
  scale_x_continuous(labels = scales::comma_format(scale = 1, suffix = ""))

bp_balance<-ggplot(bank, aes(x = "", y = balance)) +
  geom_boxplot(fill="steelblue") +
  labs(title = "Boxplot di balance",
       x = "",
       y = "balance")

grid.arrange(hist_balance, bp_balance, ncol = 2)


# ----

#credit_score
hist_score<-ggplot(bank, aes(x = credit_score, fill=churn)) + 
  geom_histogram(binwidth = 50, color = "black", position = "identity", alpha = 0.7) +
  scale_fill_manual(values = c("steelblue", "orange")) +
  labs(title = "Distribuzione Credit Score")

bp_score<-ggplot(bank, aes(x = "", y = credit_score)) +
  geom_boxplot(fill="steelblue") +
  labs(title = "Boxplot di credit_score",
       x = "",
       y = "credit_score")

grid.arrange(hist_score, bp_score, ncol = 2)


# ----

#age-balance: escludo i casi di conti vuoti pari a 0
sp1<-ggplot(subset(bank, balance != 0), aes(x = age, y = balance, color= churn)) +
  geom_point() +
  labs(title = "Scatterplot Age-balance", x = "Age", y = "balance") +
  scale_color_manual(values = c("steelblue", "orange"), name = "Churn", labels = c("No Churn", "Churn"))
sp1_density<-sp1+geom_density2d(color = "white")
sp1_density

#age-score
sp2 <- ggplot(bank, aes(x = age, y = credit_score, color = churn)) +
  geom_point() +
  labs(title = "Scatterplot age-score", x = "age", y = "score") +
  scale_color_manual(values = c("steelblue", "orange"), name = "Churn", labels = c("No Churn", "Churn"))
sp2_density <- sp2 + geom_density2d(color = "white")
sp2_density


# ----

num_data <- bank %>%
  select_if(is.numeric)

corrplot(cor(num_data), type = "upper", tl.srt = 45, tl.cex = 0.7, method = "number")


# ----


set.seed(1)
idx<-createDataPartition(bank$churn, p=0.7)
train=bank[idx$Resample1,]
test=bank[-idx$Resample1,]



# ----

dim(train)


# ----

dim(test)


# ----

bank.fit <- glm(formula = churn ~ .,
                family = binomial(link = "logit"),
                data = train
                )

summary(bank.fit) 


# ----

bank.fit2<- glm(churn ~ country+gender+age+tenure+balance+active_member,
                family = binomial(link="logit"),
                data = train)

summary(bank.fit2)


# ----

anova(bank.fit, bank.fit2, test = "LRT")


# ----

BIC(bank.fit,bank.fit2)  
AIC(bank.fit,bank.fit2)


# ----

bank.fit.clog <- glm(churn ~ .,
                     family = binomial(link = "cloglog"), 
                     data = train)

summary(bank.fit.clog)

bank.fit.clog.2<- glm(churn~country+gender+age+tenure+balance+active_member, 
                      family = binomial(link = "cloglog"), 
                      data = train)


# ----

anova(bank.fit.clog, bank.fit.clog.2, test = "LRT")


# ----

bank.fit.probit <- glm(churn ~ ., 
                       data = train ,
                       family = binomial(link="probit"))

summary(bank.fit.probit)

bank.fit.probit2 <- glm(churn ~ country+gender+age+balance+active_member, 
                        data = train ,
                        family = binomial(link="probit"))


# ----

anova(bank.fit.probit, bank.fit.probit2, test = "LRT")


# ----

matrix_AIC_BIC<-matrix(c(AIC(bank.fit2),AIC(bank.fit.probit2),AIC(bank.fit.clog.2),BIC(bank.fit2),BIC(bank.fit.probit2),BIC(bank.fit.clog.2)),2,3,byrow=TRUE)
rownames(matrix_AIC_BIC) <- c("AIC", "BIC")
colnames(matrix_AIC_BIC) <- c("logit","probit","loglog")
matrix_AIC_BIC


# ----

#coefficienti e relativi intervalli di confidenza
ci <- confint(bank.fit2)
betaHat <- coef(bank.fit2)
cbind(betaHat,ci)


# ----

#coefficients e intervalli di confidenza in termini di odds (OR)
cbind(exp(betaHat), exp(ci))


# ----

new.customers<-with(bank,
                    data.frame(gender=(c("Male","Female")),
                               country=("Germany"),
                               age=rep(round(mean(age),0)),
                               tenure=rep(round(mean(tenure),0)),
                               balance=rep(mean(balance),2),
                               active_member=(c("1","1"))))

predict(bank.fit2, newdata = new.customers, type = "response")


# ----

new.customers2<-with(bank,
                     data.frame(gender=("Male"),
                                country=("Spain"),
                                age=40,
                                tenure=5,
                                balance=rep(mean(balance),2),
                                active_member=(c("0","1"))))

predict(bank.fit2, newdata = new.customers2, type = "response")


# ----

new.customers3 <- with(bank,
                      data.frame(gender = rep(c("Male", "Female"), 50, replace = TRUE),
                                 country = rep("Spain", 50),
                                 age = rep(35, 50),
                                 tenure = rep(6, 50),
                                 balance = runif(50, 30000, 150000),
                                 active_member = rep("1", 50)))

new.customers3$Prediction<-predict(bank.fit2, newdata = new.customers3, type = "response")

#creo un nuovo dataframe includendo gli errori standard nel precedente dataframe new.customers3
new.customers4 <- cbind(new.customers3, predict(bank.fit2, newdata = new.customers3, type = "link",
se = TRUE))

#creo un nuovo dataframe sulla base del precedente, in cui calcolo le probabilità predette tramite plogis e calcolo limiti inferiore e superiore degli standard error
new.customers5 <- within(new.customers4, {
  PredictedProb <- plogis(fit)
  LL <- plogis(fit - (1.96 * se.fit))
  UL <- plogis(fit + (1.96 * se.fit))
})

#visualizzo un grafico contenente le probabilità predette e gli intervalli di confidenza. Uso geom_ribbon per rappresentare le bande degli intervalli di confidenza
ggplot(new.customers5, aes(x = balance, y = PredictedProb))+ geom_ribbon(aes(ymin = LL,
ymax = UL, fill = gender), alpha = 0.2) + geom_line(aes(colour = gender),
size = 1)



# ----

#incremento età da 40 a 50
pred.1 <- ggpredict(bank.fit2, "age [40:50]")
plot(pred.1)


# ----

#incremento di balance
pred.4 <- ggpredict(bank.fit2, "balance[50000:90000]")
plot(pred.4)



# ----


#Accuracy
prob <- predict(bank.fit2,train,type='response') 
train_matrix <- table(train$churn, as.numeric(prob>0.5))
train_accuracy <- sum(diag(train_matrix))/sum(train_matrix) #accuracy su train set
print(train_accuracy) # 0.814

prob <- predict(bank.fit2, test, type='response')
test_matrix <- table(test$churn, as.numeric(prob > 0.5))
test_accuracy <- sum(diag(test_matrix))/sum(test_matrix) #accuracy su test set
print(test_accuracy) #0.802


# ----


#ROC
M <- predict(bank.fit2, test, type="response")
MA<- prediction(M, test$churn)
perf <- performance(MA, "tpr", "fpr")
plot(perf, colorize = TRUE)
axis(1, at = seq(0,1,0.1), tck = 1, lty = 2, col = "grey", labels = NA)
axis(2, at = seq(0,1,0.1), tck = 1, lty = 2, col = "grey", labels = NA)
abline(a=0, b= 1)

auc(test$churn,M) #0.7553. Deve essere più vicino a 1 che a 0.5


# ----


#Altre metriche: Precision, Recall ed F1-score

#Train:
train_precision <- train_matrix[2,2] / sum(train_matrix[,2]) # Precision
train_recall <- train_matrix[2,2] / sum(train_matrix[2,])    # Recall
train_f1 <- 2 * (train_precision * train_recall) / (train_precision + train_recall)  # F1-score

cat("Precision sul set di addestramento:", train_precision, "\n")
cat("Recall sul set di addestramento:", train_recall, "\n")
cat("F1-score sul set di addestramento:", train_f1, "\n")

#Test:
test_precision <- test_matrix[2,2] / sum(test_matrix[,2])    # Precision
test_recall <- test_matrix[2,2] / sum(test_matrix[2,])       # Recall
test_f1 <- 2 * (test_precision * test_recall) / (test_precision + test_recall)  # F1-score

cat("Precision sul set di test:", test_precision, "\n")
cat("Recall sul set di test:", test_recall, "\n")
cat("F1-score sul set di test:", test_f1, "\n")



# ----


#Confusion matrix
pred2 <- predict(bank.fit2,test,type="response")
cutoff_churn <- ifelse(pred2>=0.50, 1,0)
cm <- confusionMatrix(as.factor(test$churn),as.factor(cutoff_churn),positive ='1')
cm


prob <- predict(bank.fit2, test, type='response')
conf_matrix <- table(test$churn, as.numeric(prob > 0.5))
conf_matrix_percent <- round(prop.table(conf_matrix, margin = 1) * 100, 1)
print(conf_matrix_percent)



