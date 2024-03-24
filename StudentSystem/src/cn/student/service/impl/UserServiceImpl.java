package cn.student.service.impl;

import cn.student.bean.User;
import cn.student.dao.UserDao;
import cn.student.dao.impl.UserDaoImpl;
import cn.student.service.UserService;

public class UserServiceImpl implements UserService {
    private UserDao dao=new UserDaoImpl();
    @Override
    public User login(User user) {
        return dao.findUserByUsernameAndPassword(user.getUsername(),user.getPassword());
    }

    @Override
    public void addUser(User user) {
         dao.add(user);
    }
}
