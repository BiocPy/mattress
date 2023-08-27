#include "Mattress.h"

//[[export]]
void* initialize_delayed_subset(void* ptr, int32_t dim, const uint32_t* subset /** numpy */, int32_t len) {
    auto mat = reinterpret_cast<Mattress*>(ptr);
    if (dim == 0) {
        return new Mattress(tatami::make_DelayedSubset<0>(mat->ptr, std::vector<uint32_t>(subset, subset + len)));
    } else {
        return new Mattress(tatami::make_DelayedSubset<1>(mat->ptr, std::vector<uint32_t>(subset, subset + len)));
    }
}
