package cn.student.dao;

import cn.student.bean.User;

public interface UserDao {
    User findUserByUsernameAndPassword(String username, String password);
    void add(User user);
}
