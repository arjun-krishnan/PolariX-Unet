function [mean_noise_stat,std_noise_stat,status] = estimate_noise(image)

image=double(image);

length_image = numel(image);
[y_data,x_data] = hist(image(1:length_image),length([min(image(1:length_image)):max(image(1:length_image))]));

mean_noise_stat = sum(x_data.*y_data./sum(y_data));
var_noise_stat = sum(x_data.^2.*y_data./sum(y_data));
std_noise_stat = sqrt(var_noise_stat - mean_noise_stat^2);


% [par] = util_gaussFit(x_data,median_filter(y_data),0,1);
% mean_noise_fit = par(2);
% std_noise_fit = (par(3)*par(4)+par(3));
% 
% 
% if ( (max(mean_noise_fit,mean_noise_stat) - min(mean_noise_fit,mean_noise_stat)) > 5*min(std_noise_fit,std_noise_stat) )||...
%     ( abs(std_noise_fit-std_noise_stat) > 5*min(std_noise_fit,std_noise_stat) )     
%     status = 'warning';
% else
status = 'ok';

return;


