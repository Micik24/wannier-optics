#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>

using namespace std;

void runtime_warning(string msg)
{
    // print warning in standard output
    cerr << "\n\n";
    cerr << "##############################################################################\n";
    cerr << "################################# WARNING ####################################\n";
    cerr << "##############################################################################\n\n";
    cerr << msg << endl << endl;
    cerr << "############################### END-WARNING ##################################\n\n";

    // print warning to special file
    ofstream file("WARNING.txt", ios_base::app);
    if (!file.is_open()) {
        cerr << "Cannot write warnings to file." << endl;
        return ;
    }

    auto current_time = chrono::system_clock::to_time_t(chrono::system_clock::now());

    file << "\n\n";
    file << "Time: " << ctime(&current_time);
    file << "##############################################################################\n";
    file << "################################# WARNING ####################################\n";
    file << "##############################################################################\n\n";
    file << msg << endl << endl;
    file << "############################### END-WARNING ##################################\n";

    file.close();
}

#endif // UTILS_H
