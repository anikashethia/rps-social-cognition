function [pay_matrix_A, pay_matrix_B] = comp_paymatrix(data) 

% extract variables
strat_space = data.strat_space(1);
win         = data.win(1);
loss        = data.loss(1);
tie         = data.tie(1);
try
    focal_A     = data.focal_own(1);
    focal_B     = data.focal_other(1);
    add_r_A     = data.bonus_A(1);
    add_r_B     = data.bonus_B(1);
catch
    [focal_A,focal_B,add_r_A,add_r_B] = deal(NaN);
end

% construct payoff matrices
for i = 1:1:strat_space
   wins = win; 
   
   if i==1
       pay_matrix(1:strat_space,1) = tie;
       pay_matrix(i+1,1) = wins;
       pay_matrix(strat_space,1) = loss;
   elseif i >= 2 && i < strat_space
       pay_matrix(1:strat_space,i) = tie;
       pay_matrix(i-1,i) = loss;
       pay_matrix(i+1,i)= wins;       
   
   elseif i==strat_space
       pay_matrix(1:strat_space,strat_space) = tie;
       pay_matrix(1,strat_space) = wins;
       pay_matrix(strat_space-1,strat_space) = loss;
   end
         
end

pay_matrix_A=pay_matrix;
pay_matrix_B=pay_matrix;

if focal_A > 0 && focal_B > 0
    if focal_A==1
        pay_matrix_A(focal_A, strat_space) = pay_matrix(focal_A, strat_space) + add_r_A;
    else
        pay_matrix_A(focal_A, focal_A-1) = pay_matrix(focal_A, focal_A-1) + add_r_A;
    end

    if focal_B==1
        pay_matrix_B(focal_B, strat_space) =pay_matrix(focal_B, strat_space) + add_r_B;
    else
        pay_matrix_B(focal_B, focal_B-1) =pay_matrix(focal_B, focal_B-1) + add_r_B;
    end
end

end
