#ifndef LASTKRYLOVSLOTS_H
#define LASTKRYLOVSLOTS_H
#include "Vector.h"
#include <map>

namespace Dmrg {

// Maps a P-vector's primary targetVectors_ slot index to the last Krylov slot
// allocated for its time evolution.  Stored here so both Pvectors (which registers
// the slot) and TargetingCommon (which resolves |P.last> labels) can reach it
// through their shared ApplyOperatorExpression.
class LastKrylovSlots {

public:

	void registerSlot(SizeType pIndex, SizeType slotIndex) { map_[pIndex] = slotIndex; }

	int getSlot(SizeType pIndex) const
	{
		const auto it = map_.find(pIndex);
		return (it == map_.end()) ? -1 : static_cast<int>(it->second);
	}

private:

	std::map<SizeType, SizeType> map_;
};

} // namespace Dmrg
#endif // LASTKRYLOVSLOTS_H
