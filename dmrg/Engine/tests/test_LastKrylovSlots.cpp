#include "LastKrylovSlots.h"
#include "PsimagLite.h"
#include <catch2/catch_test_macros.hpp>

using Dmrg::LastKrylovSlots;

TEST_CASE("LastKrylovSlots returns -1 for unregistered index", "[LastKrylovSlots]")
{
	LastKrylovSlots reg;
	CHECK(reg.getSlot(0) == -1);
	CHECK(reg.getSlot(5) == -1);
}

TEST_CASE("LastKrylovSlots registers and retrieves a single slot", "[LastKrylovSlots]")
{
	LastKrylovSlots reg;
	reg.registerSlot(2, 7);
	CHECK(reg.getSlot(2) == 7);
}

TEST_CASE("LastKrylovSlots handles multiple P-vectors independently", "[LastKrylovSlots]")
{
	LastKrylovSlots reg;
	reg.registerSlot(0, 4);
	reg.registerSlot(1, 8);
	reg.registerSlot(3, 12);

	CHECK(reg.getSlot(0) == 4);
	CHECK(reg.getSlot(1) == 8);
	CHECK(reg.getSlot(2) == -1); // never registered
	CHECK(reg.getSlot(3) == 12);
}

TEST_CASE("LastKrylovSlots overwrites on re-registration", "[LastKrylovSlots]")
{
	LastKrylovSlots reg;
	reg.registerSlot(0, 4);
	reg.registerSlot(0, 9);
	CHECK(reg.getSlot(0) == 9);
}
