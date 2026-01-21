#pragma once

#include <thread>
#include <vector>
#include <functional>

namespace threading
{
    // Basic parallel with std::thread

inline void parallel_for(std::size_t count, std::function<void(std::size_t)> func)
{
    if (count==0) return;

    if (count == 1)
    {
        func(0);
        return;
    }

    std::vector<std::thread> threads; //list to hold thread objects
    threads.reserve(count);  //tells vector how much memory to grab upfront (no need to resize later to allocate more space)

    for (std::size_t i = 0; i < count; ++i )
    {
        // emplace (build std::thread object at vec's memory)
        // [&func,i] - pass &func by reference and i by value
        threads.emplace_back([&func,i]() 
        {
            func(i); 
        }); 
    }

    for(auto& t : threads)
    {
        t.join(); // dont return yet until every thread has finished
    }



}

} // namespace threading