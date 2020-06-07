#include <stdlib.h>
#include <string.h>
#include "ParOptOptions.h"

ParOptOptions::ParOptOptions(){
  iter = entries.begin();
}

ParOptOptions::~ParOptOptions(){}

/**
  Add an string option to the set of options.

  A string option may take any valid string value.

  @param name The name of the option, must be unique
  @param value The default value of the option
  @param descript The description of the option
*/
int ParOptOptions::addStringOption( const char *name,
                                    const char *value,
                                    const char *descript ){
  int fail = 1;
  if (name && descript && entries.count(name) == 0){
    ParOptOptionEntry *entry = new ParOptOptionEntry();

    // Set the entry type
    entry->type_info = PAROPT_STRING_OPTION;

    // Copy the name of the option and its description
    entry->name = new char[ strlen(name)+1 ];
    strcpy(entry->name, name);

    entry->descript = new char[ strlen(descript)+1 ];
    strcpy(entry->descript, descript);

    entry->str_value = NULL;
    entry->str_default = NULL;
    if (value){
      entry->str_value = new char[ strlen(value)+1 ];
      strcpy(entry->str_value, value);

      entry->str_default = new char[ strlen(value)+1 ];
      strcpy(entry->str_default, value);
    }

    // Specify the entry type
    entries[name] = entry;
    fail = 0;
  }

  return fail;
}

/**
  Add a boolean option to the set of options

  A boolean option may be either true or false, represented by
  an integer.

  @param name The name of the option, must be unique
  @param value The default value of the option
  @param descript The description of the option
*/
int ParOptOptions::addBoolOption( const char *name,
                                  int value,
                                  const char *descript ){
  int fail = 1;
  if (name && descript && entries.count(name) == 0){
    ParOptOptionEntry *entry = new ParOptOptionEntry();

    // Set the entry type
    entry->type_info = PAROPT_BOOLEAN_OPTION;

    // Copy the name of the option and its description
    entry->name = new char[ strlen(name)+1 ];
    strcpy(entry->name, name);

    entry->descript = new char[ strlen(descript)+1 ];
    strcpy(entry->descript, descript);

    entry->bool_value = value;
    entry->bool_default = value;

    // Specify the entry type
    entries[name] = entry;
    fail = 0;
  }

  return fail;
}

/**
  Add an integer option to the set of options

  An integer option may take any values within the specified range
  including the end points. Default values specified outside of the
  range are truncated to their nearest limit.

  @param name The name of the option, must be unique
  @param default_value The default value of the option
  @param low The lower value for the integer range
  @param high The higher value for the integer range
  @param descript The description of the option
*/
int ParOptOptions::addIntOption( const char *name,
                                 int value,
                                 int low, int high,
                                 const char *descript ){
  int fail = 1;
  if (name && descript && entries.count(name) == 0){
    ParOptOptionEntry *entry = new ParOptOptionEntry();

    // Set the entry type
    entry->type_info = PAROPT_INT_OPTION;

    // Copy the name of the option and its description
    entry->name = new char[ strlen(name)+1 ];
    strcpy(entry->name, name);

    entry->descript = new char[ strlen(descript)+1 ];
    strcpy(entry->descript, descript);

    if (value < low){
      value = low;
    }
    if (value > high){
      value = high;
    }
    entry->int_value = value;
    entry->int_default = value;
    entry->int_low = low;
    entry->int_high = high;

    // Specify the entry type
    entries[name] = entry;
    fail = 0;
  }

  return fail;
}

/**
  Add an integer option to the set of options

  @param name The name of the option, must be unique
  @param default_value The default value of the option
  @param low The lower value for the integer range
  @param high The higher value for the integer range
  @param descript The description of the option
*/
int ParOptOptions::addFloatOption( const char *name,
                                   double value,
                                   double low, double high,
                                   const char *descript ){
  int fail = 1;
  if (name && descript && entries.count(name) == 0){
    ParOptOptionEntry *entry = new ParOptOptionEntry();

    // Set the entry type
    entry->type_info = PAROPT_FLOAT_OPTION;

    // Copy the name of the option and its description
    entry->name = new char[ strlen(name)+1 ];
    strcpy(entry->name, name);

    entry->descript = new char[ strlen(descript)+1 ];
    strcpy(entry->descript, descript);

    if (value < low){
      value = low;
    }
    if (value > high){
      value = high;
    }
    entry->float_value = value;
    entry->float_default = value;
    entry->float_low = low;
    entry->float_high = high;

    // Specify the entry type
    entries[name] = entry;
    fail = 0;
  }

  return fail;
}

/**
  Add an enumerated option to the set of options

  The value of this option is limited to a discrete set of string-valued
  options that are specified at the time when the option is added.

  @param name The name of the option, must be unique
  @param default_value The default value of the option
  @param low The lower value for the integer range
  @param high The higher value for the integer range
  @param descript The description of the option
*/
int ParOptOptions::addEnumOption( const char *name,
                                  const char *value,
                                  int size, const char *options[],
                                  const char *descript ){
  int fail = 1;
  if (name && entries.count(name) == 0){
    ParOptOptionEntry *entry = new ParOptOptionEntry();

    // Set the entry type
    entry->type_info = PAROPT_ENUM_OPTION;

    // Copy the name of the option and its description
    entry->name = new char[ strlen(name)+1 ];
    strcpy(entry->name, name);

    entry->descript = new char[ strlen(descript)+1 ];
    strcpy(entry->descript, descript);

    if (value){
      entry->enum_value = new char[ strlen(value)+1 ];
      strcpy(entry->enum_value, value);

      entry->enum_default = new char[ strlen(value)+1 ];
      strcpy(entry->enum_default, value);
    }

    entry->num_enum = size;
    entry->enum_range = new char*[ size ];
    for ( int i = 0; i < size; i++ ){
      entry->enum_range[i] = new char[ strlen(options[i])+1 ];
      strcpy(entry->enum_range[i], options[i]);
    }

    // Specify the entry type
    entries[name] = entry;
    fail = 0;
  }

  return fail;
}

/*
  Is this an option that has been added or not?

  @param name Name of the option
  @return Boolean value indicating if the option is valid
*/
int ParOptOptions::isOption( const char *name ){
  if (entries.count(name) > 0){
    return 1;
  }
  return 0;
}

/**
  Set either an enum or string type option

  @param name Name of the option
  @param value Value to set
*/
int ParOptOptions::setOption( const char *name,
                              const char *value ){
  int fail = 1;
  if (entries.count(name) == 0){
    fprintf(stderr, "ParOptOptions Error: Unknown option %s\n", name);
  }
  else {
    ParOptOptionEntry *entry = entries[name];
    if (entry->type_info == PAROPT_STRING_OPTION){
      if (entry->str_value){
        delete [] entry->str_value;
      }
      entry->str_value = new char[ strlen(value)+1 ];
      strcpy(entry->str_value, value);
      fail = 0;
    }
    else if (entry->type_info == PAROPT_ENUM_OPTION){
      int k = 0;
      for ( ; k < entry->num_enum; k++ ){
        if (strcmp(entry->enum_range[k], value) == 0){
          if (entry->enum_value){
            delete [] entry->enum_value;
          }
          entry->enum_value = new char[ strlen(value)+1 ];
          strcpy(entry->enum_value, value);
          fail = 0;
          break;
        }
      }
      if (k == entry->num_enum){
        fprintf(stderr, "ParOptOptions Warning: Option %s out or range "
                "for field %s\n", value, name);
      }
    }
    else {
      fprintf(stderr, "ParOptOptions Error: Option %s not found\n", name);
    }
  }
  return fail;
}

/**
  Set either an enum or string type option

  @param name Name of the option
  @param value Value to set
*/
int ParOptOptions::setOption( const char *name,
                              int value ){
  int fail = 1;
  if (entries.count(name) == 0){
    fprintf(stderr, "ParOptOptions Error: Unknown option %s\n", name);
  }
  else {
    ParOptOptionEntry *entry = entries[name];
    if (entry->type_info == PAROPT_BOOLEAN_OPTION){
      ParOptOptionEntry *entry = entries[name];
      entry->bool_value = value;
      fail = 0;
    }
    else if (entry->type_info == PAROPT_INT_OPTION){
      ParOptOptionEntry *entry = entries[name];
      if (value >= entry->int_low && value <= entry->int_high){
        entry->int_value = value;
        fail = 0;
      }
      else {
        fprintf(stderr, "ParOptOptions Warning: Integer option %s out or range\n", name);
      }
    }
    else {
      fprintf(stderr, "ParOptOptions Error: Option %s not found\n", name);
    }
  }
  return fail;
}

/**
  Set either an enum or string type option

  @param name Name of the option
  @param value Value to set
*/
int ParOptOptions::setOption( const char *name,
                              double value ){
  int fail = 1;
  if (entries.count(name) == 0){
    fprintf(stderr, "ParOptOptions Error: Unknown float option %s\n", name);
  }
  else {
    ParOptOptionEntry *entry = entries[name];
    if (entry->type_info == PAROPT_FLOAT_OPTION){
      ParOptOptionEntry *entry = entries[name];
      if (value >= entry->float_low && value <= entry->float_high){
        entry->float_value = value;
        fail = 0;
      }
      else {
        fprintf(stderr, "ParOptOptions Warning: Float option %s out or range\n", name);
      }
    }
    else {
      fprintf(stderr, "ParOptOptions Error: Option %s is not a float\n", name);
    }
  }
  return fail;
}

const char* ParOptOptions::getStringOption( const char *name ){
  if (entries.count(name) != 0){
    ParOptOptionEntry *entry = entries[name];
    if (entry->type_info == PAROPT_STRING_OPTION){
      return entry->str_value;
    }
  }
  fprintf(stderr, "ParOptOptions Error: String option %s not found\n", name);
  return NULL;
}

int ParOptOptions::getBoolOption( const char *name ){
  if (entries.count(name) != 0){
    ParOptOptionEntry *entry = entries[name];
    if (entry->type_info == PAROPT_BOOLEAN_OPTION){
      return entry->bool_value;
    }
  }
  fprintf(stderr, "ParOptOptions Error: Boolean option %s not found\n", name);
  return 0;
}

int ParOptOptions::getIntOption( const char *name ){
  if (entries.count(name) != 0){
    ParOptOptionEntry *entry = entries[name];
    if (entry->type_info == PAROPT_INT_OPTION){
      return entry->int_value;
    }
  }
  fprintf(stderr, "ParOptOptions Error: Integer option %s not found\n", name);
  return 0;
}

double ParOptOptions::getFloatOption( const char *name ){
  if (entries.count(name) != 0){
    ParOptOptionEntry *entry = entries[name];
    if (entry->type_info == PAROPT_FLOAT_OPTION){
      return entry->float_value;
    }
  }
  fprintf(stderr, "ParOptOptions Error: Float option %s not found\n", name);
  return 0.0;
}

const char* ParOptOptions::getEnumOption( const char *name ){
  if (entries.count(name) != 0){
    ParOptOptionEntry *entry = entries[name];
    if (entry->type_info == PAROPT_ENUM_OPTION){
      return entry->enum_value;
    }
  }
  fprintf(stderr, "ParOptOptions Error: Boolean option %s not found\n", name);
  return NULL;
}

int ParOptOptions::getOptionType( const char *name ){
  if (entries.count(name) != 0){
    return entries[name]->type_info;
  }
  fprintf(stderr, "ParOptOptions Error: Option %s not found\n", name);
  return 0;
}

const char* ParOptOptions::getDescription( const char *name ){
  if (entries.count(name) != 0){
    return entries[name]->descript;
  }
  fprintf(stderr, "ParOptOptions Error: No description for option %s found\n", name);
  return NULL;
}

int ParOptOptions::getIntRange( const char *name,
                                int *low, int *high ){
  int fail = 1;
  if (entries.count(name) == 0){
    fprintf(stderr, "ParOptOptions Error: Unknown option %s\n", name);
  }
  else {
    ParOptOptionEntry *entry = entries[name];
    if (entry->type_info == PAROPT_INT_OPTION){
      ParOptOptionEntry *entry = entries[name];
      if (low){
        *low = entry->int_low;
      }
      if (high){
        *high = entry->int_high;
      }
      fail = 0;
    }
    else {
      fprintf(stderr, "ParOptOptions Error: %s is not an integer option\n",
              name);
    }
  }
  return fail;
}

int ParOptOptions::getFloatRange( const char *name,
                                  double *low, double *high ){
  int fail = 1;
  if (entries.count(name) == 0){
    fprintf(stderr, "ParOptOptions Error: Unknown option %s\n", name);
  }
  else {
    ParOptOptionEntry *entry = entries[name];
    if (entry->type_info == PAROPT_FLOAT_OPTION){
      ParOptOptionEntry *entry = entries[name];
      if (low){
        *low = entry->float_low;
      }
      if (high){
        *high = entry->float_high;
      }
      fail = 0;
    }
    else {
      fprintf(stderr, "ParOptOptions Error: %s is not a float option\n",
              name);
    }
  }
  return fail;
}

int ParOptOptions::getEnumRange( const char *name,
                                 int *size, const char *const **values ){
  int fail = 1;
  if (entries.count(name) == 0){
    fprintf(stderr, "ParOptOptions Error: Unknown option %s\n", name);
  }
  else {
    ParOptOptionEntry *entry = entries[name];
    if (entry->type_info == PAROPT_ENUM_OPTION){
      ParOptOptionEntry *entry = entries[name];
      if (size){
        *size = entry->num_enum;
      }
      if (values){
        *values = entry->enum_range;
      }
      fail = 0;
    }
    else {
      fprintf(stderr, "ParOptOptions Error: %s is not an enum option\n",
              name);
    }
  }
  return fail;
}

void ParOptOptions::printSummary( FILE *fp, int output_level ){
  if (fp){
    std::map<std::string, ParOptOptionEntry*>::iterator it = entries.begin();

    // Iterate over the map using Iterator till end.
    while (it != entries.end()){
      ParOptOptionEntry *entry = it->second;

      if (output_level > 0){
        fprintf(fp, "%s\n", entry->descript);
      }
      if (entry->type_info == PAROPT_STRING_OPTION){
        fprintf(fp, "%-40s %-15s\n",
                entry->name, entry->str_value);
        if (output_level > 0){
          fprintf(fp, "%-40s %-15s\n", "default", entry->str_default);
        }
      }
      else if (entry->type_info == PAROPT_BOOLEAN_OPTION){
        fprintf(fp, "%-40s %-15d\n",
                entry->name, entry->bool_value);
        if (output_level > 0){
          fprintf(fp, "%-40s %-15d\n", "default", entry->bool_default);
        }
      }
      else if (entry->type_info == PAROPT_INT_OPTION){
        fprintf(fp, "%-40s %-15d\n",
                entry->name, entry->int_value);
        if (output_level > 0){
          fprintf(fp, "%-40s %-15d\n", "default", entry->int_default);
        }
      }
      else if (entry->type_info == PAROPT_FLOAT_OPTION){
        fprintf(fp, "%-40s %-15g\n",
                entry->name, entry->float_value);
        if (output_level > 0){
          fprintf(fp, "%-40s %-15g\n", "default", entry->float_default);
        }
      }
      else if (entry->type_info == PAROPT_ENUM_OPTION){
        fprintf(fp, "%-40s %-15s\n",
                entry->name, entry->enum_value);
        if (output_level > 0){
          for ( int i = 0; i < entry->num_enum; i++ ){
            fprintf(fp, "%-40s %-15s\n", "options", entry->enum_range[i]);
          }
          fprintf(fp, "%-40s %-15s\n", "default", entry->enum_default);
        }
      }
      if (output_level > 0){
        fprintf(fp, "\n");
      }
      it++;
    }
  }
}

void ParOptOptions::begin(){
  iter = entries.begin();
}

const char* ParOptOptions::getName(){
  if (iter != entries.end()){
    return iter->first.c_str();
  }
  return NULL;
}

int ParOptOptions::next(){
  if (iter != entries.end()){
    iter++;
  }
  return (iter != entries.end());
}