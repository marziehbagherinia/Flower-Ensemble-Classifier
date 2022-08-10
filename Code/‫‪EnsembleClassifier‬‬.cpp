#include<iostream>
#include <vector>
#include <fstream>
#include <string>
#include <bits/stdc++.h>
using namespace std;

void readClassifiers(vector< vector< vector< float >>> &classifiers, string wDirectory)
{
    string line, field, firstLine;
    vector< vector< float >> helperV;  
    vector< float > gettingLine;
    int i = 0;
    while(true)
    {
        string index = to_string(i);
        string name = "./" + wDirectory + "/classifier_" + index + ".csv"; 
        ifstream myfile(name.c_str());
        if(myfile.is_open())
        {
            getline(myfile, firstLine);
            while (getline(myfile,line))
            {
                gettingLine.clear();
                stringstream ss(line);
                while(getline(ss,field,','))
                {
                    gettingLine.push_back(stof(field));
                }
                helperV.push_back(gettingLine); 
            }
            classifiers.push_back(helperV);
            helperV.clear();
            i++;
        }
        else 
            break;
    }       
}

void readInstance(vector< vector< float >> &features, string vDirectory)
{
    string line, field, firstLine;  
    vector<float> helperV;
    string name = "./" + vDirectory + "/dataset.csv";
    ifstream myfile(name.c_str(), ios::out);

    if (myfile.is_open())
    {
        getline(myfile, firstLine);
        while (getline(myfile,line))
        {
            helperV.clear();
            stringstream ss(line);   
            while(getline(ss,field,','))
            {
                helperV.push_back(stof(field));
            }
            features.push_back(helperV); 
        }
    }
}

vector< int > readLabels(vector< int > labels, string vDirectory)
{
    string line, firstLine;
    string name = "./" + vDirectory + "/labels.csv";
    ifstream file(name.c_str());

    if (file.is_open())
    {
        getline(file,firstLine);
        while (getline(file, line)) 
        {
            labels.push_back(stoi(line));
        }
    }
    return labels;
}

vector< vector< float >> linearClassification( vector< vector< vector< float >>>classifiers , vector< float > instance)
{
    vector< vector< float >> dClassification(classifiers.size(), vector< float >(classifiers[0].size()));
    
    for (int i = 0; i < classifiers.size(); ++i)
    {
        for (int j = 0; j < classifiers[i].size(); ++j)
         {
            dClassification[i][j]=classifiers[i][j][0]*instance[0] + 
            classifiers[i][j][1]*instance[1] + classifiers[i][j][2];
         } 
    }
    return dClassification;
}

vector< int > findClass( vector< vector< float >> dClassification)
{
    vector< float > mValue(dClassification.size());
    vector< int > iClass(dClassification.size());

    for (int i = 0; i < mValue.size(); ++i)
    {
        mValue[i] = *max_element(dClassification[i].begin(), dClassification[i].end());
    }

    for (int i = 0; i < dClassification.size(); ++i)
    {
        for (int j = 0; j < dClassification[i].size(); ++j)
        {
            if (mValue[i] == dClassification[i][j])
            {
                iClass[i]=j;
                break;
            }
            continue;
        }
    }
    return iClass; 
}

int EnsembleClassification( vector< int > iClass)
{
    int max = *max_element(iClass.begin(), iClass.end());
    vector< int > counter(max+1);

    for (int i = 0; i < counter.size(); ++i)
    {
        counter[i] = count(iClass.begin(), iClass.end(), i);
    }

    int max1 = *max_element(counter.begin(), counter.end());    
    for (int i = 0; i < counter.size(); ++i)
    {
        if (counter[i] == max1)
        {
            return i;
            break;
        }
        continue;
    }
}

vector<int> loops(vector< vector< vector< float >>> classifiers, vector< vector< float >> features)
{
    vector< vector< vector< float >>> helperV;
    for (int i = 0; i < features.size(); ++i)
    {
        helperV.push_back(linearClassification(classifiers, features[i]));
    }

    vector< vector< int >> iClass;
    for (int i = 0; i < helperV.size(); ++i)
    {
        iClass.push_back(findClass(helperV[i]));
    }

    vector<int> Olabels;
    for (int i = 0; i < iClass.size(); ++i)
    {
        Olabels.push_back(EnsembleClassification(iClass[i]));
    }
    return Olabels; 
}

float coAccuracy(vector< int > Ilabels, vector< int > Olabels)
{
    float checker = 0;

    for (int i = 0; i < Ilabels.size(); ++i)
    {
        if (Olabels[i] == Ilabels[i])
        {
            checker++;
        }
        continue;
    }
    
    float accuracy = checker/10;
    accuracy = floor(100*accuracy)/100;
    return accuracy;    
}


float computing(vector< vector< vector< float >>> classifiers, vector< vector< float >> features, 
                vector< int > labels, string wDirectory, string vDirectory)
{
    readClassifiers(classifiers, wDirectory);
    readInstance(features, vDirectory);
    vector< int > Ilabels = readLabels(labels, vDirectory);
    vector< int > Olabels = loops(classifiers, features);
    float accuracy = coAccuracy(Ilabels, Olabels);
    return accuracy;
}

int main(int argc, char *argv[])
{
    string vDirectory = argv[1];
    string wDirectory = argv[2];
    vector< vector< vector< float >>> classifiers;
    vector< vector< float >> features;
    vector< int > labels;
    float accuracy = computing(classifiers, features, labels, wDirectory, vDirectory);

    printf("Accuracy: %.2f%c\n", accuracy, '%');
}