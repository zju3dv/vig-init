find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

set(VIGINIT_EXAMPLE_SOURCES
    opencv_image.cpp
    euroc_data_stream.cpp
    main.cpp
)

add_executable(viginit-example
    ${VIGINIT_EXAMPLE_SOURCES}
)

target_include_directories(viginit-example
    PRIVATE
        ${YAML_CPP_INCLUDE_DIR}
)

target_link_libraries(viginit-example
    PRIVATE
        viginit
        ${OpenCV_LIBS}
        ${YAML_CPP_LIBRARIES}
)
