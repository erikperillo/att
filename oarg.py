"""
COPYRIGHT (C) 2015, Erik Perillo
All rights reserved.

Redistribution and use in source and binary forms, 
with or without modification, are permitted provided that the 
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its 
contributors may be used to endorse or promote products derived from this 
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys
import re

class InvalidKeyNameFormat(Exception):
    pass

class InvalidOptionsPassed(Exception):
    pass

class UnknownKeyword(Exception):
    def __init__(self, message, key):
        super(UnknownKeyword, self).__init__(message)
        self.key = pure_name(key)

class RepeatedKeyError(Exception):
    pass

def pure_name(name):
    while name.startswith("-"):
        name = name.replace("-", "", 1)
    return name

def cmd_line_key(name):
    return ("-" if len(name) <= 1 else "--") + name

def is_single_dash_key(name, digits="0123456789"):
    return name.startswith("-") and not (any(name.startswith(d, 1) \
           for d in digits) or name.startswith("-", 1))

def is_double_dash_key(name):
    return name.startswith("--")

def is_key(name, digits="0123456789"):
    return is_single_dash_key(name, digits) or is_double_dash_key(name)

def cond_split(tgt_list, function):
    cands = []
    remaining = []
    i = 0

    while i < len(tgt_list):
        if function(tgt_list[i]):
            cand = [tgt_list[i]]
            i += 1
            while i < len(tgt_list) and not function(tgt_list[i]):
                cand.append(tgt_list[i])
                i += 1
            cands.append(cand)
        else:
            remaining.append(tgt_list[i])
            i += 1

    return cands, remaining

def tokenize(tgt_list, delim=","):
    semi_tokens = (2*delim).join(tgt_list)

    while 3*delim in semi_tokens:
        semi_tokens = semi_tokens.replace(3*delim, delim)

    #default string: r"(?<!\\)(?:\\\\)*(?<!,),(?!,)"
    semi_tokens = re.split(r"(?<!\\)(?:\\\\)*(?<!" + delim + r")" + delim + 
                           r"(?!" + delim + r")", semi_tokens) 

    tokens = [[s.replace("\\" + delim, delim) for s in st.split(2*delim)] \
             for st in semi_tokens]

    return tokens 

oargs = []
unknown_keys = []

def reset():
    global oargs, unknown_keys
    oargs = []
    unknown_keys = []

class Oarg:
    def __init__(self, keywords, def_val, description, pos_n_found=-1, 
                 single=True):
        list_keywords = keywords.split() if isinstance(keywords, str) \
                        else keywords

        #checking errors
        for key in list_keywords:
            if not is_key(key):
                raise InvalidKeyNameFormat(("invalid format for '%s': " % key) +
                                            " must be of the form '-e'" +
                                            " or '--example'")
            if is_single_dash_key(key) and len(key) > 2:
                raise InvalidKeyNameFormat(("invalid format for key '%s': " % \
                                           key) + "only one letter allowed " + 
                                                  "for single dash")
            if is_double_dash_key(key) and len(key) <= 3:
                raise InvalidKeyNameFormat(("invalid format for key '%s': " % \
                                            key) + "only two or more letters" +
                                                   " allowed for double dash")
            if any(key in oarg.keywords for oarg in oargs):
                raise RepeatedKeyError("key '%s' already exists" % \
                                       cmd_line_key(key))

        #assigning attributes
        self.tp           = type(def_val)
        self.keywords     = [pure_name(n) for n in list_keywords]
        self.def_val      = def_val
        self.description  = description
        self.pos_n_found  = pos_n_found
        self.found        = False
        self.vals         = (def_val,)
        self.single       = single

        #putting in list
        oargs.append(self)

    @property
    def val(self):
        return self.vals[0]    

    def setVals(self, str_vals, falses):
        if self.tp is bool:
            self.vals = tuple(not val in falses for val in str_vals)
        else:
            self.vals = tuple(self.tp(val) for val in str_vals)
    
def parse(source=sys.argv[1:], delim=",", 
          falses=["false", "no", "n", "not", "0"]):
    global unknown_keys
    bool_oargs = [oarg for oarg in oargs if oarg.tp is bool]

    candidates, remaining = cond_split(source, is_key)
    remaining_tokens = tokenize(remaining, delim)
    
    for candidate_block in candidates:
        name, str_values = candidate_block[0], candidate_block[1:]

        #multiple boolean values
        if is_single_dash_key(name) and len(name) > 2:
            for option in name.replace("-", "", 1):
                try:
                    oarg = filter(lambda oarg: option in oarg.keywords, 
                                  bool_oargs)[0]
                except IndexError:
                    unknown_keys.append(pure_name(name))
                    continue

                oarg.vals = (not oarg.def_val,)
                oarg.found = True
        else:
            try:
                oarg = filter(lambda oarg: pure_name(name) in oarg.keywords, 
                              oargs)[0]
            except IndexError:
                unknown_keys.append(pure_name(name))
                continue

            if not str_values:
                if oarg.tp is bool:
                    oarg.vals = (not oarg.def_val,)
                else:
                    raise InvalidOptionsPassed("no argument provided for '%s'"\
                                               % name)
            else:
                tokens = tokenize(str_values, delim)
                
                if oarg.single:
                    values = [tokens[0][0]]
                    remaining_tokens += [tokens[0][1:]] + tokens[1:]
                else:
                    values = tokens[0]
                    remaining_tokens += tokens[1:]

                oarg.setVals(values, falses)

            oarg.found = True

    _remaining_oargs = [oarg for oarg in oargs \
                        if not oarg.found and oarg.pos_n_found >= 0]
    remaining_oargs = sorted(_remaining_oargs, key=lambda o: o.pos_n_found)

    #black magic begins here
    for oarg in remaining_oargs:
        remaining_tokens = filter(lambda tok: tok and tok != [""], 
                                  remaining_tokens)
        if not remaining_tokens:
            break

        if oarg.single:
            values = [remaining_tokens[0][0]]
            remaining_tokens = [remaining_tokens[0][1:]] + remaining_tokens[1:]
        else:
            values = remaining_tokens[0]
            remaining_tokens = remaining_tokens[1:]

        oarg.setVals(values, falses)
        oarg.found = True

    if unknown_keys:
        key = unknown_keys[0]
        raise UnknownKeyword("unknown key '%s' passed" % cmd_line_key(key), 
                             pure_name(key))

def describe_args(helpmsg="", def_val=False, min_spacing=16):
    max_width = max(sum(len(name) for name in oarg.keywords) + \
                2*(len(oarg.keywords)-1) for oarg in oargs)

    if helpmsg:
        print helpmsg

    for oarg in oargs:
        keywords = ", ".join([cmd_line_key(n) for n in oarg.keywords])
        print ("{0:" + str(max_width + min_spacing) + "}{1}").format(keywords,
               oarg.description) + (" ({})".format(oarg.def_val) \
                                   if def_val else "")

#legacy reasons
describeArgs = describe_args

if __name__ == "__main__":
    ival = Oarg("-i --intval", 34, "Integer value", 1, True)
    ival2 = Oarg("-j --intval2", 34, "Integer value", 10, True)
    ival3 = Oarg("-k --intval3", 34, "Integer value", 5, False)
    fval = Oarg("-f --floatval", -34.034, "Float value", 3, True)
    fval2 = Oarg("-g --floatval2", -3.034, "Float value", 0, False)
    fval3 = Oarg("-h --floatval3", 0.034, "Float value", 6, True)
    sval = Oarg("-s --strval", "ay lmao", "String value", 4, True)
    sval2 = Oarg("-t --strval2", "ay lmaojdfh", "String value", 9, False)
    sval3 = Oarg("-u --strval3", "ay lmaodfh", "String value", 8, False)
    bval = Oarg("-b --boolval", False, "Boolean value", 2, False)
    bval2 = Oarg("-c --boolval2", True, "Boolean value", 7, False)
    bval3 = Oarg("-d --boolval3", False, "Boolean value", 8, True)
    hlp = Oarg("-H --help", False, "This help message", 10)

    parse()

    if hlp.val:
        describe_args("Available args", True)
        exit()

    print "founds:"
    for o in ival, ival2, ival3, fval, fval2, fval3, sval, \
             sval2, sval3, bval, bval2, bval3:
        if o.found:
            print "keywords:", o.keywords, "vals:", o.vals
            print "defval:", o.def_val, "val:", o.val
            print
    
