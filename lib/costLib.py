import numpy as np
import sys
import time

class cost_function_info_for_jada:
    def __init__(self, inputfilename):
        # Read in quadratic values
        self.inputfilename = inputfilename
        iter = self.__read_iteration_number("Quadratic cost function: J ")
        J = self.__read_file("Quadratic cost function: J ")
        JoJc = self.__read_file("Quadratic cost function: JoJc")
        Jb = self.__read_file("Quadratic cost function: Jb")
        J_nl = self.__read_file("CostFunction: Nonlinear")
        self.nobs = self.__read_nobs("Nonlinear Jo(")
        # Deal with old and new builds
        if iter[0] == 0:
            self.J = []
            self.JoJc = []
            self.Jb = []
            self.iter = []
        elif iter[0] == 1:
            self.J = [J_nl[0]]
            self.JoJc = [J_nl[0]]
            self.Jb = [0.0]
            self.iter = [0]
        else:
            raise ValueError('iteration vector for jada is incorrect', iter)
        # Add data to object
        for aa in range(0, len(J)):
            self.J.append(J[aa])
            self.JoJc.append(JoJc[aa])
            self.Jb.append(Jb[aa])
            self.iter.append(iter[aa])

    def __read_file(self, query_string):
        infile = open(self.inputfilename, 'r')
        vals = []
        for line in infile:
            if query_string in line:
                value = line.rstrip().split("=")[1]
                vals.append(float(value))
        infile.close()
        return np.ravel(vals)

    def __read_iteration_number(self, query_string):
        infile = open(self.inputfilename, 'r')
        vals = []
        for line in infile:
            if query_string in line:
                start_index = line.find("(")
                end_index = line.find(")")
                value = int(line[start_index + 1:end_index])
                vals.append(float(value))
        infile.close()
        return np.ravel(vals)

    def __read_nobs(self, query_string):
        infile = open(self.inputfilename, 'r')
        varname = []
        nobs = []
        for line in infile:
            if query_string in line:
                # Get variable name
                start_index = line.find("(")
                end_index = line.find(")")
                value = line[start_index + 1:end_index]
                varname.append(value)
                # Get number os obs
                str1 = line.split("nobs")[1]
                str2 = str1.split(", Jo/n")[0]
                str3 = str2.split("=")[1]
                value = int(str3)
                nobs.append(value)
        infile.close()
        dictionary = dict(zip(varname, nobs))
        sum_values = sum(dictionary.values())
        return sum_values

class cost_function_info_for_var:

    def __init__(self, inputfilename):
        self.inputfilename = inputfilename
        file_contents = self.__read_varstat_file(inputfilename)
        header_dict = self.__get_header_info(file_contents)
        lines_to_read = int(np.ceil(float(header_dict['evaluations']) / 10.0))
        stats_dict = self.__get_penalty_stats(file_contents, lines_to_read)
        self.nobs = self.__get_total_nobs(file_contents)
        self.J = stats_dict["penalty"]
        self.JoJc = stats_dict["obspenalty"]
        self.Jb = stats_dict["bgpenalty"]
        self.iter = stats_dict["niter"]

    def __read_varstat_file(self, filename):
        f = open(filename, 'r')
        file_content = f.read()
        f.close()
        lines = file_content.replace('>', '').split('\n')
        return lines

    def __get_header_info(self, lines):
        for iline in range(0, len(lines)):
            if "year month day hour minute  iterations evaluations Nstatistics" in lines[iline]:
                headers = lines[iline].split()
                values = lines[iline + 1].split()
        d = {}
        for aa in range(0, len(headers)):
            d[headers[aa]] = values[aa]
        return d

    def __read_values(self, filelines, firstline, lastline):
        array = filelines[firstline].split()
        for aa in range(firstline + 1, lastline):
            array.extend(filelines[aa].split())
        return array

    def __read_section(self, lines, iline, num_read, factor):
        # niter
        firstline = iline + 1
        lastline = firstline + num_read
        niter = self.__read_values(lines, firstline, lastline)
        niter = np.array(niter[1:], np.dtype(np.int16))
        # neval
        firstline = lastline
        lastline = firstline + num_read
        # J
        firstline = lastline
        lastline = firstline + num_read
        penalty = self.__read_values(lines, firstline, lastline)
        penalty = np.array(penalty, np.dtype(np.float32))
        penalty = penalty * factor

        return niter, penalty

    def __get_penalty_stats(self, lines, num_read):
        d = {}
        for iline in range(0, len(lines)):
            if ": Total penalty" in lines[iline]:
                factor = lines[iline].split(":")[0]
                factor = float(factor.split("/")[1])
                niter, penalty = self.__read_section(lines, iline, num_read, factor)
                d["penalty"] = penalty
                d["niter"] = niter
            if ": Obs penalty" in lines[iline]:
                factor = lines[iline].split(":")[0]
                factor = float(factor.split("/")[1])
                niter, obspenalty = self.__read_section(lines, iline, num_read, factor)
                d["obspenalty"] = obspenalty
            if ": Bg penalty" in lines[iline]:
                if "each" not in lines[iline]:
                    factor = lines[iline].split(":")[0]
                    factor = float(factor.split("/")[1])
                    niter, bgpenalty = self.__read_section(lines, iline, num_read, factor)
                    d["bgpenalty"] = bgpenalty
        return d

    def __get_total_nobs(self, lines):
        start_reading = False
        sum_value = 0
        for line in lines:
            if "COUNT MEAN" in line:
                factor = float(line.split("/")[1].split(":")[0])
                start_reading = True
                local_sum = 0
                continue
            if line == '' and start_reading:
                #("Local sum = ", local_sum)
                start_reading = False
            if start_reading:
                strarray = line.split()
                local_sum += int(float(strarray[-1]) * factor)
                sum_value += int(float(strarray[-1]) * factor)
        return sum_value
