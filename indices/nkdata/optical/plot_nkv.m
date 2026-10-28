
function plot_nkv(File_List, colors, titulo)


for a = 1:size(File_List,1),

   Table = read_nkv (deblank(File_List(a,:))) ;
   l   = Table(:,1);              % in nm
   n   = Table(:,2);
   k   = Table(:,3);

   %%%Pintamos
   subplot (2,1,1);
   plot (l,n, colors(a), 'Linewidth', 2);
   hold on;
   
   subplot (2,1,2);
   plot (l,k, colors(a), 'Linewidth', 2);
   hold on;
   
end;

subplot (2,1,1);
title (titulo, 'Fontweight', 'bold', 'Fontsize', 12);
xlabel ('Lambda [nm]', 'Fontweight', 'bold');
ylabel ('n', 'Fontweight', 'bold');
grid on;
legend (File_List,'Location','Best');
set (gcf, 'Position', [230 180 550 480]);

subplot (2,1,2);
xlabel ('Lambda [nm]', 'Fontweight', 'bold');
ylabel ('k', 'Fontweight', 'bold');
grid on;
legend (File_List,'Location','Best');
set (gcf, 'Position', [230 180 550 480]);
