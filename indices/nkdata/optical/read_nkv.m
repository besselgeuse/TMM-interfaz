function  nk_Table = read_nkv (file_nkv);
%
%  read_nkv.M
%  ******************************************************************
%
%  Este programa matlab pretende leer los ficheros de constantes ópticas
%  con formato nkv (formato ASCII propio del GS35)
%
%  La salida es una tabla (nk_Table) que tienes tres columnas: la primera 
%  con lambda en nm, la segunda con n y la tercera con k
%
%
%  Versión 1.0
%  12 - 8 - 2010
%
% ----------


% Abrimos el archivo con la medida. Primero obviamos los datos
% de la cabecera que comienzan por ";"

File_Id = fopen (file_nkv,'r');
FilePos = ftell(File_Id);
Linea   = fgetl(File_Id);

while (Linea(1)==';'),
   FilePos = ftell(File_Id);
   Linea   = fgetl(File_Id);
end

% Hemos leido el primer dato numérico, así que recolocamos el puntero de
% lectura del fichero
hayerror = fseek(File_Id,FilePos, 'bof');

% Luego leemos los datos en la variable Temp, cuya primera
% columna contiene las lambda, la segunda el índice de refracción (n) y la
% tercera el coeficiencte de extinción (k)
% La variable Cuantos refleja el número de elementos leidos con
% éxito. Sabiendo que hay tres columnas, Cuantos/3 es el número
% de elementos por columna.

[Temp_Rows, Cuantos] = fscanf (File_Id, '%f', [3,10000]);
nk_Table = Temp_Rows';
fclose (File_Id);
