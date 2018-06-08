#include "catch.hpp"

// the things to set up a test ingredients
#include <LeMonADE/core/Ingredients.h>
#include <LeMonADE/feature/FeatureMoleculesIO.h>
#include <LeMonADE/feature/FeatureAttributes.h>
#include <LeMonADE/feature/FeatureExcludedVolumeSc.h>
#include <LeMonADE/feature/FeatureFixedMonomers.h>

#include "GPUScBFM_AB_Type.h"

TEST_CASE( "Information from ingredients are passed to GPU Class","[ingredients]" ){

    // set up the basic ingredients
    typedef LOKI_TYPELIST_3( FeatureMoleculesIO, FeatureAttributes, FeatureExcludedVolumeSc<> ) Features;
    typedef ConfigureSystem<VectorInt3,Features,4> Config;
    typedef Ingredients < Config> IngredientsType;
    IngredientsType myIngredients;

    int box=32;
    myIngredients.setBoxX(box);
    myIngredients.setBoxY(box);
    myIngredients.setBoxZ(box);
    myIngredients.setPeriodicX(true);
    myIngredients.setPeriodicY(true);
    myIngredients.setPeriodicZ(true);
    myIngredients.modifyBondset().addBFMclassicBondset();
    myIngredients.modifyMolecules().addMonomer(0,0,0);
    myIngredients.modifyMolecules().addMonomer(2,0,0);
    myIngredients.modifyMolecules().connect(0,1);
    myIngredients.modifyMolecules().addMonomer(4,0,0);
    myIngredients.modifyMolecules().connect(1,2);
    myIngredients.modifyMolecules().addMonomer(4,2,0);
    myIngredients.modifyMolecules().connect(2,3);
    myIngredients.modifyMolecules().addMonomer(4,4,0);
    myIngredients.modifyMolecules().connect(3,4);
    myIngredients.modifyMolecules().addMonomer(6,0,0);
    myIngredients.modifyMolecules().connect(2,5);

    // check system integrity
    REQUIRE_NOTHROW(myIngredients.synchronize());

    SECTION( "Constructor call with 2 arguments" ) {
        CHECK(myIngredients.getBoxX() == box);
        //GPUScBFM_AB_Type<IngredientsType> myGPUScBFM(myIngredients,1000);
        //CHECK(myGPUScBFM.readIngredients().getBoxX() == box);
    }

}