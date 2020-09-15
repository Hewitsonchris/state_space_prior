library(data.table)
library(ggplot2)

rm(list = ls())

d1 <- fread('../fits/fit_individual_1.txt')
d2 <- fread('../fits/fit_individual_2.txt')
d3 <- fread('../fits/fit_individual_3.txt')
d4 <- fread('../fits/fit_individual_4.txt')
d5 <- fread('../fits/fit_individual_5.txt')

d <- list(d1, d2, d3, d4, d5)

for(i in 1:length(d)) {
  setnames(d[[i]], c('alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'base_fb', 'w',
                     'gamma_ff', 'gamma_fb', 'gamma_fb2', 'sse'))
  d[[i]][, group := i]
}

d <- rbindlist(d)

dd <- melt(d, id.vars='group')
dd <- dd[variable != 'sse']
dd <- dd[group != 3]
dd <- dd[variable %in% c('gamma_ff', 'gamma_fb', 'gamma_fb2')]
ddd <- dd[, .(mean(value), 1.96 * sd(value) / sqrt(.N)), .(group, variable)]

ggplot(ddd, aes(factor(group), V1)) +
  geom_hline(yintercept=0) +
  geom_pointrange(aes(ymin=V1-V2, ymax=V1+V2)) +
  facet_wrap(~variable, ncol=3, scales='free') +
  theme_classic() +
  theme(aspect.ratio = 1)

dd[, t.test(value), .(group, variable)][order(group)]
