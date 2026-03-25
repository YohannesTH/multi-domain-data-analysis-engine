% Module 4: Healthcare Analysis in MATLAB (PCA)
disp('Loading healthcare data...');
data = readtable('../../data/healthcare_data_scaled.csv');

% The data is already scaled in Python
X = table2array(data);

disp('Performing PCA for dimensionality reduction...');
[coeff, score, latent, tsquared, explained] = pca(X);

% Display variance explained by first two components
fprintf('Variance explained by Principal Component 1: %.2f%%\n', explained(1));
fprintf('Variance explained by Principal Component 2: %.2f%%\n', explained(2));
fprintf('Total variance explained by first 2 PCs: %.2f%%\n', sum(explained(1:2)));

% Visualize clusters
disp('Visualizing PCA clusters...');
figure('Position', [100, 100, 800, 600]);
scatter(score(:,1), score(:,2), 25, 'filled', 'MarkerFaceAlpha', 0.6);
title('PCA of Healthcare Dataset (First 2 Components)');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
grid on;

% Save output plot
if ~exist('../../output', 'dir')
    mkdir('../../output');
end
saveas(gcf, '../../output/module4_healthcare_pca.png');
disp('Saved visualization to output/module4_healthcare_pca.png');

% Statistical validation (e.g., correlations of PC1 with original features)
disp('Statistical validation: Correlation of original features with PC1');
[~, sorted_idx] = sort(abs(coeff(:,1)), 'descend');
fprintf('Top 5 features contributing to PC1 (indices): %s\n', num2str(sorted_idx(1:5)'));

exit;
