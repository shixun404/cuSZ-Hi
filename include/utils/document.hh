/**
 * @file document.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-22
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef ARGUMENT_PARSER_DOCUMENT_HH
#define ARGUMENT_PARSER_DOCUMENT_HH

#include <regex>
#include <string>


const std::string fmt_b("\e[1m");
const std::string fmt_0("\e[0m");

const std::regex  bful("@(.*?)@");
const std::string bful_text("\e[1m\e[4m$1\e[0m");
const std::regex  bf("\\*(.*?)\\*");
const std::string bf_text("\e[1m$1\e[0m");
const std::regex  ul(R"(_((\w|-|\d|\.)+?)_)");
const std::string ul_text("\e[4m$1\e[0m");
const std::regex  red(R"(\^\^(.*?)\^\^)");
const std::string red_text("\e[31m$1\e[0m");

std::string  //
Format(const std::string& s)
{
    auto a = std::regex_replace(s, bful, bful_text);
    auto b = std::regex_replace(a, bf, bf_text);
    auto c = std::regex_replace(b, ul, ul_text);
    auto d = std::regex_replace(c, red, red_text);
    return d;
}

static const char psz_short_doc[] =
    // "cusz, version [placeholder]\n"
    "\n"
    "usage: cuszhi [-zxrh] [-i file] [-t dtype] [-m mode] [-e eb] [-l x,y,z] [--predictor NAME] [-a preference] [-s Lossless_pipeline]"
    "...\n"
    "\n"
    "  z : zip/compress\n"
    "  x : unzip/decompress\n"
    "  r : dryrun\n"
    "  h : print full-length help document (not up-to-date)\n"
    "\n"
    "  i file  : path to input datum\n"
    "  t dtype : f32 or fp4 (to be updated)\n"
    "  m mode  : compression mode; abs, r2r\n"
    "  e eb    : error bound; default 1e-4\n"
    "  l size  : \"-l x\" for 1D; \"-l [X]x[Y]\" for 2D; \"-l [X]x[Y]x[Z]\" for 3D\n"
    "  predictor : (default) spline, lorenzo\n"
    "  a preference: cr-first (default, better compression ratio) or rd-first (better rate-distortion)\n"
    "  s Lossless_pipeline: cr (default, high-ratio but slow) or tp (fast but lower CR)\n"
    // "  p pred  : select predictor from \"lorenzo\" and \"spline3d\"\n"
    "\n"
    "  report list: \n"
    "      syntax: opt[=v], \"kw1[=(on|off)],kw2[=(on|off)]\n"
    "      keyworkds: time, quality\n"
    "      example: \"--report time\", \"--report time=off\"\n"
    "\n"
    "example:\n"
    "   ## using default Spline predictor\n"
    "   cuszhi -t f32 -m r2r -e [ErrorBound] -i [/PATH/TO/DATA] -l [X]x[Y]x[Z] -z --report time\n"
    "   cuszhi -i [/PATH/TO/DATA].cusza -x --report time --compare [/PATH/TO/DATA]\n"
    "\n"
    "   ## using Lorenzo predictor for comparison\n"
    "   cuszhi -t f32 -m r2r -e [ErrorBound] -i [/PATH/TO/DATA] -l [X]x[Y]x[Z] -z --report time -- predictor lorenzo\n"
    "   cuszhi -i [/PATH/TO/DATA].cusza -x --report time --compare [/PATH/TO/DATA]\n";

static const char psz_full_doc[] = //todo: revise
    "*NAME*\n"
    "        cuSZ: CUDA-Based Error-Bounded Lossy Compressor for Scientific Data\n"
    "        Lowercased \"*cusz*\" is the command.\n"
    "\n"
    "*SYNOPSIS*\n"
    "        The basic use is listed below,\n"
    "        *cusz* *-t* f32 *-m* r2r *-e* 1.0e-4.0 *-i* ./data/cesm-CLDHGH-3600x1800 *-l* 3600,1800 *-z* *--report* "
    "time\n"
    //   cusz -t f32 -m r2r -e 1.0e-4.0 -i ./data/cesm-CLDHGH-3600x1800 -l 3600x1800 -z --report time\n
    "             ^^------ ------ ----------- ------------------------------- ------------  |  ^^\n"
    "             ^^ dtype  mode  error bound            input file           low-to-high  zip ^^\n"
    "\n"
    "        *cusz* *-i* ./data/cesm-CLDHGH-3600x1800.cusza *-x* *--compare* ./data/cesm-CLDHGH-3600x1800 *--report* "
    "time\n"
    //       cusz -i ./data/cesm-CLDHGH-3600x1800.cusza -x --compare ./data/cesm-CLDHGH-3600x1800 --report
    //       time\n"
    "             ^^-------------------------------------  |   ^^\n"
    "             ^^            compressed file          unzip ^^\n"
    "\n"
    "        *cusz* *-t* f32|64 *-m* [eb mode] *-e* [eb] *-i* [datum file] *-l* [x[,y[,z]]] *-z*\n"
    "        *cusz* *-i* [basename].cusza *-x*\n"
    "\n"
    "*OPTIONS*\n"
    "    *Mandatory* (zip and dryrun)\n"
    "        *-z* or *--compress* or *--*@z@*ip*\n"
    "        *-r* or *--dry-*@r@*un*\n"
    "                No lossless Huffman codec. Only to get data quality summary.\n"
    "                In addition, quant. rep. and dict. size are retained\n"
    "\n"
    "        *-m* or *--*@m@*ode* <abs|r2r>\n"
    "                Specify error-controlling mode. Supported modes include:\n"
    "                _abs_: absolute mode, eb = input eb\n"
    "                _r2r_: relative-to-value-range mode, eb = input eb x value range\n"
    "\n"
    "        *-e* or *--eb* or *--error-bound* [num]\n"
    "                Specify error bound. e.g., _1.23_, _1e-4_, _1.23e-4.56_\n"
    "\n"
    "        *-i* or *--*@i@*nput* [file]\n"
    "\n"
    "        *-d* or *--dict-size* [256|512|1024|...]\n"
    "                Specify dictionary size/quantization bin number.\n"
    "                Should be a power-of-2.\n"
    "\n"
    "        *-l* [x[,y[,z]]]   Specify (1|2|3)D data size, with dimensions from low to high.\n"
    "\n"
    "    *Mandatory* (unzip)\n"
    "        *-x* or *--e*@x@*tract* or *--decompress* or *--unzip*\n"
    "\n"
    "        *-i* or *--*@i@*nput* [corresponding datum basename (w/o extension)]\n"
    "\n"
    "    *Additional*\n"
    "        *-p* or *--*@p@*redictor*\n"
    "                Select predictor from \"lorenzo\" (default) or \"spline3d\" (3D only).\n"
    "        *--origin* or *--compare* /path/to/origin-datum\n"
    "                For verification & get data quality evaluation.\n"
    "        *--opath*  /path/to\n"
    "                Specify alternative output path.\n"
    "\n"
    "    *Modules*\n"
    "        *--skip* _module-1_,_module-2_,...,_module-n_,\n"
    "                Disable functionality modules. Supported module(s) include:\n"
    "                _huffman_  Huffman codec after prediction+quantization (p+q) and before reversed p+q.\n"
    "                _write2disk_  Skip write decompression data.\n"
    //    "\n"
    //    "        *-p* or *--pre* _method-1_,_method-2_,...,_method-n_\n"
    //    "                Enable preprocessing. Supported preprocessing method(s) include:\n"
    //    "                _binning_  Downsampling datum by 2x2 to 1.\n"
    "\n"
    "    *Print Report to stdout*\n"
    "        *--report* (option=on/off)-list\n"
    "                Syntax: opt[=v], \"kw1[=(on|off)],kw2=[=(on|off)]\n"
    "                Keyworkds: time  quality  compressibility\n"
    "                Example: \"--report time\", \"--report time=off\"\n"
    "\n"
    "    *Demonstration*\n"
    "        *-h* or *--help*\n"
    "                Get help documentation.\n"
    "\n"
    //    "        *-V* or *--verbose*\n"
    //    "                Print host and device information for diagnostics.\n"
    //    "\n"
    //    "        *-M* or *--meta*\n"
    //    "                Get archive metadata. (TODO)\n"
    "\n"
    "    *Advanced Runtime Configuration*\n"
    "        *--demo* [demo-dataset]\n"
    "                Use demo dataset, will omit given dimension(s). Supported datasets include:\n"
    "                1D: _hacc_  _hacc1b_    2D: _cesm_  _exafel_\n"
    "                3D: _hurricane_  _nyx-s_  _nyx-m_  _qmc_  _qmcpre_  _rtm_  _parihaka_\n"
    "\n"
    "        *-c* or *--config* (option=value)-list\n"
    "               Syntax: opt=v, \"kw1=val1,kw1=val2[,...]\"\n"
    "                   + *eb*=<val>    error bound\n"
    "                   + *cap*=<val>   capacity, number of quant-codes\n"
    "                   + *demo*=<val>  skip length input (\"-l x[,y[,z]]\"), alternative to \"--demo dataset\"\n"
    "\n"
    // "               Other internal parameters:\n"
    // "                   + *quantbyte*=<1|2>\n"
    // "                       Specify quantization code representation.\n"
    // "                       Options _1_, _2_ are for *1-* and *2-*byte, respectively. (default: 2)\n"
    // "                       ^^Manually specifying this may not result in optimal memory footprint.^^\n"
    // "                   + *huffbyte*=<4|8>\n"
    // "                       Specify Huffman codeword representation.\n"
    // "                       Options _4_, _8_ are for *4-* and *8-*byte, respectively. (default: 4)\n"
    // "                       ^^Manually specifying this may not result in optimal memory footprint.^^\n"
    // "                   + *huffchunk*=[256|512|1024|...]\n"
    // "                       Manually specify chunk size for Huffman codec, overriding autotuning.\n"
    // "                       Should be a power-of-2 that is sufficiently large.\n"
    // "                       ^^This affects Huffman decoding performance significantly.^^\n"
    // "\n"
    "*EXAMPLES*\n"
    "    *Demo Datasets*\n"
    "        Set a *shell variable*:\n"
    "        export PATH=$(pwd)/bin:$PATH\n"
    "        CESM=./data/cesm-CLDHGH-3600x1800\n"
    "        HURR=./data/hurr-CLOUDf48-500x500x100\n"
    "\n"
    "        *CESM* example:\n"
    "        cusz -t f32 -m r2r -e 1e-4 -i ${CESM} -l 3600x1800 -z --report time\n"
    "        cusz -t f32 -m r2r -e 1e-4 -i ${CESM} -l 3600x1800 -r\n"
    "        cusz -i ${CESM}.cusza -x --report time --compare ${CESM} --skip write2disk\n"
    "\n"
    "        *CESM* example with specified output path:\n"
    "        mkdir data2 data3\n"
    "        ^^# zip, output to `data2`^^\n"
    "        cusz -t f32 -m r2r -e 1e-4 -i ${CESM} -l 3600x1800 -z --opath data2\n"
    "        ^^# unzip, in situ^^\n"
    "        cusz -i ${CESM}.cusza -x && ls data2\n"
    "        ^^# unzip, output to `data3`^^\n"
    "        cusz -i ${CESM}.cusza -x --opath data3 && ls data3\n"
    "        ^^# unzip, output to `data3`, compare to the original datum^^\n"
    "        cusz -i ${CESM}.cusza -x --opath data3 --compare ${CESM} && ls data3\n"
    "\n"
    "        *Hurricane Isabel* example:\n"
    "        cusz -t f32 -m r2r -e 1e-4 -i ${HURR} -l 500x500x100 -z\n"
    "        cusz -t f32 -m r2r -e 1e-4 -i ${HURR} -l 500x500x100 -r\n"
    "        cusz -i ${HURR}.cusza -x\n"
    "\n";

// TODO
// "        *EXAFEL* example:\n"
// "        cusz -t f32 -m r2r -e 1e-4 -i ./data/exafel-59200x388 --demo exafeldemo -z -x --pre binning\n"
// "        cusz -t f32 -m r2r -e 1e-4 -i ./data/exafel-59200x388 --demo exafeldemo -z -x --pre binning "
// "--skip huffman\n"
// "        cusz -i ./data/exafel-59200x388.BN.cusza -x\n";

static const char huff_re_short_doc[] =
    "\n"
    "OVERVIEW: Huffman submodule as standalone program\n"  // TODO from this line on
    "\n"
    "USAGE:\n"
    "  The basic use with demo datum is listed below,\n"
    "    ./huff --encode --decode --verify --input ./baryon_density.dat.b16 \\\n"
    "        -3 512 512 512 --input-rep 16 --huffman-rep 32 --huffman-chunk 2048 --dict-size 1024\n"
    "  or shorter\n"
    "    ./huff -e -d -V -i ./baryon_density.dat.b16 -3 512 512 512 -R 16 -H 32 -C 2048 -c 1024\n"
    "            ^  ^  ^ --------------------------- -------------- ----- ----- ------- -------\n"
    "            |  |  |       input datum file         dimension   input Huff. Huff.   codebook\n"
    "          enc dec verify                                       rep.  rep.  chunk   size\n"
    "\n"
    "EXAMPLES\n"
    "  Essential:\n"
    "    ./bin/huff -e -d -i ./baryon_density.dat.b16 -3 512 512 512 -R 16 -c 1024\n"
    "    have to input dimension, and higher dimension for a multiplication of each dim.,\n"
    "    as default values input-rep=16 (bits), huff-rep=32 (bits), codebook-size=1024 (symbols)\n"
    "\n";

static const char doc_dim_order[] =
    "\n"
    "  Input dimension follows low-to-high (e.g., x-y-z) order.\n"
    "  Taking 2D CESM-ATM as an example, \n"
    "\n"
    "  |<------------------------- x 3600 --------------------------->|    \n"
    "  +--------------------------------------------------------------+  - \n"
    "  |                                                              |  ^ \n"
    "  |                                                              |  | \n"
    "  |              CESM-ATM:    1800x3600 (y-x order)              |  | \n"
    "  |              datum name:  <field>_1800_3600                  |  y \n"
    "  |                                                              | 1800 \n"
    "  |              input:       -l 3600,1800                       |  | \n"
    "  |              input order: -l [x,y]                           |  | \n"
    "  |                                                              |  | \n"
    "  |                                                              |  v \n"
    "  +--------------------------------------------------------------+  - \n"
    "\n"
    "  Taking 3D Hurricane as another example, whose dimensions are\n"
    "  100x500x500, the input is \"-l 500,500,100\".\n";

#endif
