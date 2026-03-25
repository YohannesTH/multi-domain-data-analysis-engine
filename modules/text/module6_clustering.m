% Module 6: MATLAB Clustering on Text Features
disp('Loading text features...');
data = readtable('../../data/text_features_tfidf.csv');

% Convert table to array
X = table2array(data);

% 1. Perform Clustering
disp('Performing K-means clustering (k=3)...');
k = 3;
[idx, C] = kmeans(X, k, 'Replicates', 5, 'Distance', 'cosine');

% 2. Visualize Clusters
disp('Visualizing clusters using PCA for 2D representation...');
[coeff, score] = pca(X);

figure('Position', [100, 100, 800, 600]);
gscatter(score(:,1), score(:,2), idx, 'rgb', 'osd');
title('K-means Clustering of Text Features (k=3)');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Location', 'best');
grid on;

% Save output plot
if ~exist('../../output', 'dir')
    mkdir('../../output');
end
saveas(gcf, '../../output/module6_text_clustering.png');
disp('Saved visualization to output/module6_text_clustering.png');

% Optional: Print top terms for each cluster
disp('Top contributing features for each cluster center (Indices):');
for i = 1:k
    [~, sorted_idx] = sort(C(i,:), 'descend');
    fprintf('Cluster %d top 5 features: %s\n', i, num2str(sorted_idx(1:5)));
end

exit;
