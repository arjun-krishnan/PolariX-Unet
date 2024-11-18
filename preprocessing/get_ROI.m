function [image_roi,status,is_beam] = get_ROI(image,threshold_factor,bits,disp_choice)

if nargin<4
    disp_choice = 0;
end
if nargin<3
    bits = 12;	
end
if nargin<2
    threshold_factor = 2;
end

% convert image to double precision 
image = double(image);

%% Define parameters
status = '';	
beam_i_threshold = 0.122*2^(bits); % if bits=12,beam_i_threshold=500;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the mathematical definition of the matrix M:
% filter_size = 1.5;
% filter = fspecial('gaussian',round(filter_size*4+1),filter_size);

% M=zeros(7,7);
% for x=-3:3
%     for y=-3:3
%         M(x+4,y+4)=1/sqrt(2*pi*1.5^2)*exp(-(x^2+y^2)/2/1.5^2);
%     end
% end
% M=M/sum(sum(M));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 M = [0.0013    0.0041    0.0079    0.0099    0.0079    0.0041    0.0013
      0.0041    0.0124    0.0241    0.0301    0.0241    0.0124    0.0041
      0.0079    0.0241    0.0470    0.0587    0.0470    0.0241    0.0079
      0.0099    0.0301    0.0587    0.0733    0.0587    0.0301    0.0099
      0.0079    0.0241    0.0470    0.0587    0.0470    0.0241    0.0079
      0.0041    0.0124    0.0241    0.0301    0.0241    0.0124    0.0041
      0.0013    0.0041    0.0079    0.0099    0.0079    0.0041    0.0013];

%% Check noise level in the image
[mean_noise1,~,status_noise1] = estimate_noise(image)

if strcmp(status_noise1,'warning') 
    status = append(status,'- Noise estimate (1) failed -');
end

image = image - mean_noise1;
image_filt = conv2(image, M, 'same');
[mean_noise2,std_noise2,status_noise2] = estimate_noise(image_filt);

if strcmp(status_noise2,'warning')
    status = append(status,'- Noise estimate (2) failed -');
end

threshold = mean_noise2 + threshold_factor * std_noise2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % the functionality of the following part of codes is still under test. It
% % can be simply skipped.
% if flag1*flag2
% 	length_image_filt = numel(image_filt);
% 	[y_data,x_data] = hist(image_filt(1:length_image_filt),length([min(image_filt(1:length_image_filt)):max(image_filt(1:length_image_filt))]));
% 
% 	[par1, yFit1] = util_gaussFit(x_data,median_filter(y_data),0,1);
% 	
% 	k=0;
%     fit_start = [];
%     while isempty(fit_start);
%         fit_start = find(y_data >= par1(1)*(1000-k)/1000, 1, 'last' ) + 2;
%         k = k+1;
%     end
% 	[par2, yFit2] = util_gaussFit(x_data(fit_start:end),median_filter(y_data(fit_start:end)),0,1);
% 	mean_noise2 = par2(2);
% 	std_noise2 = (par2(3)*par2(4)+par2(3));
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Find ROI

[num_pix_x, num_pix_y] = size(image_filt); 

candidate_list = find(image_filt(1:(num_pix_x*num_pix_y))>=threshold);
num_candidates_start = length(candidate_list);

if num_candidates_start==0
    status = append(status,'- ROI not found - ');
end

roi = zeros(num_pix_x,num_pix_y);
roi(candidate_list) = 1;

image_filt_start = conv2(image_filt, ones(5,5)/5^2, 'same');
[~,start_roi_x] = max(max(image_filt_start,[],1));
[~,start_roi_y] = max(max(image_filt_start,[],2));
roi = bwselect(logical(roi),start_roi_x,start_roi_y);

image_roi = roi.*image;

% 21/03/23 new expression to set negaitve pixels to zero
%image_roi = max(image_roi,0);

% 21/03/23 this was replaced with above expression
%roi_thres_noise = image_roi < 0; % logical array for pxels < 0
%image_roi(roi_thres_noise) = 0; % set all pixels < 0 to zero

%% simple idea for determing is_beam
temp = image_filt.*roi;
int_max = max(temp(temp > 0));
int_min = min(temp(temp > 0));
int_diff = int_max - int_min;

if int_diff > beam_i_threshold
    is_beam = 1;
else
    is_beam = 0;
    status = append(status,'- Beam not found - ');
end

if disp_choice
    if ~isempty(status)
        disp(status);
    else
        disp('get ROI succesful')
    end
end

return;
