%Display data Display 2D data in a nice grid 
% [ h, display array] = displayData(X,eample_width) displays 2D data stored in a X in a nice grid . It returns the 
% figure handle h and the displayed array if requested 

function [h, display_array] = displayDataexample(X,example_width)

%disp(example_width);

% set example_width automatically if not passed in 
if ~exist('example_width','var') || isempty(example_width)
	example_width = round(sqrt(size(X,2)));
end 

disp(X);

% Gray color 

colormap(gray);

disp(example_width);

%compute rows and cols
[m n]=size(X)
example_height = (n / example_width);


%compute number of items to display
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% between image padding 

pad =1 

%setup blank display

display_array = - ones(pad + display_rows * (example_height + pad) ,pad + display_cols * (example_width + pad));


% copy each example into a patch on the display array 

curr_ex = 1;



endfunction 


