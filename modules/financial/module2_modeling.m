% Module 2: Financial Modeling in MATLAB
disp('Loading financial data...');
data = readtable('../../data/financial_data.csv');

% Prepare data
% Assuming 'Close' is the target variable for fitting
% We will fit standard polynomial models to the time series index
n = height(data);
t = (1:n)';
y = data.Close;

% 2. Perform Calculations
disp('Performing polynomial regression (Curve fitting)...');

% Fit polynomial of degree 3
p3 = polyfit(t, y, 3);
y_fit3 = polyval(p3, t);

% Fit polynomial of degree 5
p5 = polyfit(t, y, 5);
y_fit5 = polyval(p5, t);

% Error Analysis (RMSE)
rmse_3 = sqrt(mean((y - y_fit3).^2));
rmse_5 = sqrt(mean((y - y_fit5).^2));

fprintf('RMSE for Degree 3 Polynomial: %.4f\n', rmse_3);
fprintf('RMSE for Degree 5 Polynomial: %.4f\n', rmse_5);

% Plot the curves
disp('Generating Plot...');
figure('Position', [100, 100, 800, 600]);
plot(t, y, 'k.', 'DisplayName', 'Actual Close Price');
hold on;
plot(t, y_fit3, 'r-', 'LineWidth', 2, 'DisplayName', 'Degree 3 Fit');
plot(t, y_fit5, 'b--', 'LineWidth', 2, 'DisplayName', 'Degree 5 Fit');
title('Polynomial Curve Fitting of Stock Close Prices');
xlabel('Time Index');
ylabel('Price');
legend('Location', 'best');
grid on;

% Save output plot
if ~exist('../../output', 'dir')
    mkdir('../../output');
end
saveas(gcf, '../../output/module2_curve_fitting.png');
disp('Saved visualization to output/module2_curve_fitting.png');

% Note: Comparison with Python can be done by inspecting RMSE and visualizing the
% difference between Python ARIMA/LR logic and pure polynomial interpolation here.
exit;
