#ifndef PAR_OPT_OPTIONS_H
#define PAR_OPT_OPTIONS_H

#include "ParOptVec.h"
#include <string>
#include <map>

class ParOptOptions : public ParOptBase {
 public:
  static const int PAROPT_STRING_OPTION = 1;
  static const int PAROPT_BOOLEAN_OPTION = 2;
  static const int PAROPT_INT_OPTION = 3;
  static const int PAROPT_FLOAT_OPTION = 4;
  static const int PAROPT_ENUM_OPTION = 5;

  ParOptOptions();
  ~ParOptOptions();

  // Add the options
  int addStringOption( const char *name,
                       const char *value,
                       const char *descript );
  int addBoolOption( const char *name, int value,
                     const char *descript );
  int addIntOption( const char *name, int value,
                    int low, int high,
                    const char *descript );
  int addFloatOption( const char *name, double value,
                      double low, double high,
                      const char *descript );
  int addEnumOption( const char *name, const char *value,
                     int size, const char *options[],
                     const char *descript );

  // Is this an option?
  int isOption( const char *name );

  // Set the option values
  int setOption( const char *name, const char *value );
  int setOption( const char *name, int value );
  int setOption( const char *name, double value );

  // Retrieve the option values that have been set
  const char* getStringOption( const char *name );
  int getBoolOption( const char *name );
  int getIntOption( const char *name );
  double getFloatOption( const char *name );
  const char* getEnumOption( const char *name );

  // Get the type
  int getOptionType( const char *name );

  // Get the description
  const char* getDescription( const char *name );

  // Get information about the range of possible values
  int getIntRange( const char *name, int *low, int *high );
  int getFloatRange( const char *name, double *low, double *high );
  int getEnumRange( const char *name, int *size,
                    const char *const **values );

  void printSummary( FILE *fp, int output_level );

  void begin();
  const char* getName();
  int next();

 private:
  class ParOptOptionEntry {
   public:
    ParOptOptionEntry(){
      name = descript = NULL;
      str_value = NULL;
      bool_value = bool_default = 0;
      int_value = int_default = int_low = int_high = 0;
      float_value = float_default = float_low = float_high = 0.0;
      num_enum = 0;
      enum_value = enum_default = NULL;
      enum_range = NULL;
      is_set = 0;
    }
    ~ParOptOptionEntry(){
      if (name){ delete [] name; }
      if (descript){ delete [] descript; }
      if (str_value){ delete [] str_value; }
      if (str_default){ delete [] str_default; }
      if (enum_value){ delete [] enum_value; }
      if (enum_default){ delete [] enum_default; }
      if (enum_range){
        for ( int i = 0; i < num_enum; i++ ){
          delete [] enum_range[i];
        }
      }
    }

    // Name of the option
    char *name;

    // Name of the description
    char *descript;

    // Flag to indicate whether the option has been set
    // by the user/algorithm
    int is_set;

    // Type of entry
    int type_info;

    // String value
    char *str_value, *str_default;

    // Set the boolean value
    int bool_value, bool_default;

    // Store information about the integer value
    int int_value, int_default, int_low, int_high;

    // Store information about the float values
    double float_value, float_default, float_low, float_high;

    // Store a list of options as strings
    int num_enum;
    char *enum_value, *enum_default, **enum_range;
  };

  std::map<std::string, ParOptOptionEntry*> entries;
  std::map<std::string, ParOptOptionEntry*>::iterator iter;
};

#endif // PAR_OPT_OPTIONS_H
