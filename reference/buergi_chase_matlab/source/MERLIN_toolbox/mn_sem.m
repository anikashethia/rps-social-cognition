function sem = mn_sem(data)

sem = std(data)/sqrt(size(data,1));

end