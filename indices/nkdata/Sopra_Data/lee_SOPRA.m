
function TablaS = lee_SOPRA(filename)

File_Id = fopen (filename,'r'); %abre el fichero en modo “lectura”
% Luego leemos los datos en la variable “Temp”, cuya primera
% columna contiene el num de la linea, la segunda contiene lambda, la tercera contiene n, la cuarta contiene k
  
File_Id = fopen (filename,'r');

Temp = textscan(File_Id,'%*s%*s%f32%f32%f32','delimiter','*','headerlines',3,'expChars','e');
%on ne veut pas lire le premier string (DATA) ni le deuxième (numéro de la
%ligne) les délimitations entre les données sont des * et non pas des
%espaces, on ne lit pas les 3 premières lignes et les exponentiels sont
%écris comme e et non pas e+ comme c'est par défaut dans Matlab


Cuantos = size(Temp{1},1)-3; % nombre de données lues dans la première cellule moins les 3 erronnées de la fin
% La variable “Cuantos” refleja el número de elementos leidos con
% éxito.

TempCortada1 = Temp{1}; 
TempCortada2 = Temp{2};
TempCortada3 = Temp{3};

fclose (File_Id);

    TablaSt(1:Cuantos,1) = TempCortada1(1:Cuantos,:);
    TablaSt(1:Cuantos,2) = TempCortada2(1:Cuantos,:);
    TablaSt(1:Cuantos,3) = TempCortada3(1:Cuantos,:);
    
    q=1;
     for p=1:Cuantos, 
        if (200<TablaSt(p,1)) && (TablaSt(p,1)<1200)
            
        TablaS(q,1)=TablaSt(p,1);
        TablaS(q,2)=TablaSt(p,2);
        TablaS(q,3)=TablaSt(p,3);
        q=q+1;
        else
            q=q;
    end;
     end;
    if q~=1
     TablaS; 
    else
        TablaS(1,:)=NaN;
    end;

    
