#include <cassert>
#include <iostream>
#include <random>
#include <string>

static bool ends_with_mtx(const std::string& s) {
    return s.size() >= 4 && s.compare(s.size() - 4, 4, ".mtx") == 0;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: ./blockcg <mode=cg,bcg,pbcg> <custom matrix>\n");
        return 1;
    }

    // if (ends_with_mtx(argv[1])) {
    //     // File mode: ./spmm <file.mtx> [feat_size]
    //     A_csr = load_mtx(argv[1], M_int, K_int);
    //     N = (argc >= 3) ? static_cast<size_t>(std::stoi(argv[2])) : 128;
    // }

    // TODO load matrix (hardcoded for now, or .mtx)
    // TODO write fn for and run cg
    // TODO write fn for and run bcg
    // TODO compare cg, bcg

    // TODO if time write fn for preconditioned bcg
    // TODO compare cg, bcg, pbcg
}
