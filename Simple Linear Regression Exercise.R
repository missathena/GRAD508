data <- read.csv("height_data_fake.csv")

height <- data$Height_feet
age <- data$Age_months
bio <- data$Biological_sex
birth <- data$Height_birth_feet

bio[which(bio=="M")] <- 0
bio[which(bio=="F")] <- 1
bio <- as.numeric(bio)

#simple linear regression
plot(age,height)
summary(lm(height ~ age))

#multiple linear regression
model <- lm(height ~ age + birth + bio)
summary(model)

