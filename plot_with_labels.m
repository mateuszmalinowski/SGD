% draw a set of 2D points with different colors for each label
% 
% Souce code from: http://lear.inrialpes.fr/src/jsgd/
% 

function col = plot_with_labels(x, labels)
  colormap = 'bgrcmy';
  dots = '.o+*';
  l = length(colormap);  

  nclass = max(labels);
  
  for k = 1:nclass
    subset = find(labels == k);
    col = [colormap(mod(k - 1, l) + 1) dots(floor((k - 1) / l) + 1)];
    plot(x(1, subset), x(2, subset), col);
    hold on
  end    
    
end
