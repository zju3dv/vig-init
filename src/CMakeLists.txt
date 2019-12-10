add_subdirectory(../ext/slamtools ${CMAKE_CURRENT_BINARY_DIR}/ext/slamtools)

set(VIGINIT_LIB_SOURCES
    imu_initializer.cpp
    vig_initializer.cpp
)

add_library(viginit
    ${VIGINIT_LIB_SOURCES}
)

target_include_directories(viginit
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(viginit
    PUBLIC
        cxx_std_17
)

target_compile_options(viginit
    PRIVATE
        -pedantic
        $<$<CONFIG:Debug>:-Wextra>
        #$<$<NOT:$<CONFIG:Debug>>:-O2>
)

target_link_libraries(viginit
    PUBLIC
        slamtools
)
